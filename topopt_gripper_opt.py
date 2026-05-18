from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
import pyamg
import scipy.spatial as spatial
from matplotlib import colors
import matplotlib.pyplot as plt
from mma import mmasub
from domain_config import get_hex_nut_elements


# MAIN DRIVER
def main(nelx, nely, volfrac, penal, rmin, ft):
    print("Inverter mechanism design with MMA")
    print("ndes: " + str(nelx) + " x " + str(nely))
    print("volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal))
    print("Filter method: " + ["Sensitivity based", "Density based"][ft])

    # Max and min stiffness
    Emin = 1e-9
    Emax = 100.0
    kspring = 1.0
    dof_in = 0

    # Fetch passive elements from external file
    hex_nut_config = get_hex_nut_elements(nelx, nely, sym_x=True)
    solid = hex_nut_config['solid']
    void = hex_nut_config['void']

    load_bottom = np.array(hex_nut_config['load_bottom'], dtype=int)
    dof_out_dist = load_bottom

    # Total degrees of freedom
    ndof = 2 * (nelx + 1) * (nely + 1)

    # Allocate design variables, initialize and bounds
    x = volfrac * np.ones(nely * nelx, dtype=float)
    xmin = np.zeros(nely * nelx)
    xmax = np.ones(nely * nelx)

    # Enforce passive element domains
    xmin[solid] = 0.999
    xmax[solid] = 1.0
    x[solid] = 1.0

    xmin[void] = 0.0
    xmax[void] = 0.001
    x[void] = 0.0

    xold1 = x.copy()
    xold2 = x.copy()
    xPhys = x.copy()
    low = np.zeros((nely * nelx, 1))
    upp = np.ones((nely * nelx, 1))

    # Element stiffness matrix
    KE = lk()

    # --- OPTIMIZATION 1: VECTORIZED EDOFMAT ASSEMBLY ---
    # Replaces the original double for-loop over (nelx, nely) with a fully
    # vectorized meshgrid approach — O(nelx*nely) work with no Python loops.
    ely, elx = np.meshgrid(np.arange(nely), np.arange(nelx))
    ely, elx = ely.flatten(), elx.flatten()
    n1 = (nely + 1) * elx + ely
    n2 = (nely + 1) * (elx + 1) + ely
    edofMat = np.column_stack([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 2*n2, 2*n2+1, 2*n1, 2*n1+1])

    # Construct base index pointers for the coo format
    iK_base = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK_base = np.kron(edofMat, np.ones((1, 8))).flatten()

    # --- OPTIMIZATION 2: VECTORIZED FILTER ASSEMBLY VIA KDTREE ---
    # Replaces the original quadruple nested for-loop filter build with a
    # cKDTree sparse distance query — from O(nelx*nely*(2*ceil(rmin))^2)
    # Python iterations down to a single vectorized C-level call.
    #
    # BUG FIX: sparse_distance_matrix already includes self-pairs (i==j) at
    # distance 0.0, giving weight rmin-0=rmin. The original attempt appended
    # the diagonal a second time, doubling every self-weight and corrupting Hs.
    # Solution: use the output as-is with no manual diagonal append.
    y_coord, x_coord = np.meshgrid(np.arange(nely), np.arange(nelx))
    coords = np.column_stack((x_coord.flatten(), y_coord.flatten()))
    tree = spatial.cKDTree(coords)

    sparse_dist = tree.sparse_distance_matrix(tree, rmin, output_type='coo_matrix')
    iH = sparse_dist.row
    jH = sparse_dist.col
    sH = np.maximum(0.0, rmin - sparse_dist.data)
    # Self-pairs (dist=0) already included above — do NOT append diagonal again.

    H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
    Hs = np.array(H.sum(1))[:, 0]

    # Boundary conditions and supports
    dofs = np.arange(ndof)

    # Base standard constraints (anchoring the bottom-left corner)
    fixed_base = np.array([2*(nely+1)-2, 2*(nely+1)-1])

    # Constrain vertical Y-displacements along the top symmetry cut-line (y=0)
    symmetry_nodes = np.arange(0, (nelx + 1) * (nely + 1), nely + 1)
    symmetry_y_dofs = 2 * symmetry_nodes + 1

    fixed = np.union1d(fixed_base, symmetry_y_dofs)
    free = np.setdiff1d(dofs, fixed)

    # Solution, RHS, and adjoint dummy load vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    adj = np.zeros((ndof, 1))

    f[dof_in, 0] = 1.0

    dobjdu = np.zeros((ndof, 1))
    dobjdu[load_bottom, 0] = 1.0 / len(load_bottom)

    # --- OPTIMIZATION 3: PRE-ASSEMBLE SPRING INDEX ARRAYS OUTSIDE LOOP ---
    # Spring DOF index/data arrays are topology-constant; build them once and
    # concatenate with the per-iteration element data inside the loop.
    iK_springs = np.concatenate(([dof_in], dof_out_dist))
    jK_springs = np.concatenate(([dof_in], dof_out_dist))
    sK_springs = np.concatenate(([kspring], np.full(len(dof_out_dist), kspring / len(dof_out_dist))))

    iK = np.concatenate((iK_base, iK_springs))
    jK = np.concatenate((jK_base, jK_springs))

    loop = 0
    change = 1
    dv = np.ones(nely * nelx)
    dobj = np.ones(nely * nelx)

    # Recalculate target volume to account for passive element presets
    active_elements = (nelx * nely) - len(solid) - len(void)
    target_volume = (volfrac * active_elements) + len(solid)

    print("Starting optimization iterations...")

    while change > 0.01 and loop < 200:
        loop += 1

        # Evaluate current element stiffness values
        sK_elements = ((KE.flatten()[np.newaxis]).T * (Emin + xPhys**penal * (Emax - Emin))).flatten(order='F')
        sK = np.concatenate((sK_elements, sK_springs))

        # Assemble and reduce global stiffness matrix
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = K[free, :][:, free]

        # --- OPTIMIZATION 4: ALGEBRAIC MULTIGRID SOLVER ---
        # Replaces spsolve (direct, O(n^1.5) for 2-D) with PyAMG's Ruge-Stüben
        # AMG (near-linear for elliptic problems). The hierarchy is rebuilt each
        # iteration so coarse-level operators always match the current K — reusing
        # a stale hierarchy degrades the preconditioner and causes more Krylov
        # iterations, making it slower and potentially non-convergent.
        ml = pyamg.ruge_stuben_solver(K)

        u[free, 0] = ml.solve(f[free, 0].flatten(), tol=1e-8)
        adj[free, 0] = ml.solve(-dobjdu[free, 0].flatten(), tol=1e-8)

        # BUG FIX: Objective must NOT be negated here.
        # MMA minimizes obj; the adjoint sensitivities already carry the correct
        # sign. Negating obj inverts the problem — MMA would minimize displacement
        # instead of maximizing it.
        obj = np.sum(u[load_bottom, 0]) / len(load_bottom)

        # Element sensitivity calculation
        dobj[:] = (penal * xPhys**(penal - 1) * (Emax - Emin)) * (
            np.dot(adj[edofMat].reshape(nelx * nely, 8), KE) * u[edofMat].reshape(nelx * nely, 8)
        ).sum(1)
        dv[:] = np.ones(nely * nelx)

        # BUG FIX: Restore the original broadcasting pattern with explicit
        # [np.newaxis].T / Hs column-vector shapes. The shorthand dobj/Hs
        # broadcasts incorrectly against the CSC matrix H, raising a ValueError.
        if ft == 0:
            dobj[:] = np.asarray((H * (x * dobj)[np.newaxis].T) / Hs)[:, 0] / np.maximum(0.001, x)
        elif ft == 1:
            dobj[:] = np.asarray(H * (dobj[np.newaxis].T / Hs))[:, 0]
            dv[:]   = np.asarray(H * (dv[np.newaxis].T / Hs))[:, 0]

        # Volume constraint normalised against corrected target volume
        vol_constraint_val = np.sum(xPhys) / target_volume - 1.0

        # MMA optimisation step
        m = 1
        a0mma, amma, cmma, dmma = 1, np.zeros((m, 1)), 1000 * np.ones((m, 1)), np.zeros((m, 1))
        (xnew, _, _, _, _, _, _, _, _, low, upp) = mmasub(
            m, nely * nelx, loop,
            x[np.newaxis].T, xmin[np.newaxis].T, xmax[np.newaxis].T,
            xold1[np.newaxis].T, xold2[np.newaxis].T,
            obj, dobj[np.newaxis].T,
            np.array([[vol_constraint_val]]), dv[np.newaxis],
            low, upp, a0mma, amma, cmma, dmma
        )

        xold2[:] = xold1
        xold1[:] = x
        x[:xnew.shape[0]] = xnew[:, 0]

        # Density filter update
        if ft == 0:
            xPhys[:] = x
        elif ft == 1:
            xPhys[:] = np.asarray(H * x[np.newaxis].T / Hs)[:, 0]

        xPhys[solid] = 1.0
        xPhys[void] = 0.0

        # Convergence check (inf-norm)
        change = np.linalg.norm(x.reshape(nelx * nely, 1) - xold1.reshape(nelx * nely, 1), np.inf)

        print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(
            loop, obj, np.sum(xPhys) / (nelx * nely), change))

    print("Optimization complete. Rendering final symmetric layout...")

    fig, ax = plt.subplots()
    xPlot = np.round(xPhys).reshape((nelx, nely))
    xPlot = np.hstack([np.fliplr(xPlot), xPlot])
    ax.imshow(-xPlot.T, cmap='gray', interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
    plt.show()
    input("Press any key to close...")


# Element stiffness matrix
def lk():
    E = 1.0
    nu = 0.3
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8, -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    KE = E / (1 - nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ])
    return KE


if __name__ == "__main__":
    main(nelx=180, nely=60, volfrac=0.5, rmin=4.0, penal=3.0, ft=1)