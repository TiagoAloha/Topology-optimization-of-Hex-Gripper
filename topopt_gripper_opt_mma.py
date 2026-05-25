from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from matplotlib import colors
import matplotlib.pyplot as plt
from domain_config import get_hex_nut_elements
from numba import njit, prange
import time

# Import Svanberg's MMA subproblem solver directly
from mma import mmasub

# ----------------------------------------------------------------------------
# Optional fast solver
# ----------------------------------------------------------------------------
try:
    from pypardiso import spsolve as pardiso_solve
    HAS_PARDISO = True
except ImportError:
    HAS_PARDISO = False


# ----------------------------------------------------------------------------
# JIT-compiled FE kernels
# ----------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def fast_assembly(xPhys, penal, Emin, Emax, KE, sK_elements):
    num_elem = xPhys.shape[0]
    for i in prange(num_elem):
        E_val = Emin + (xPhys[i] ** penal) * (Emax - Emin)
        for r in range(8):
            for c in range(8):
                sK_elements[i * 64 + r * 8 + c] = KE[r, c] * E_val


@njit(cache=True, parallel=True)
def fast_sensitivities(xPhys, penal, Emin, Emax, u, adj, edofMat, KE, dobj):
    num_elem = xPhys.shape[0]
    for i in prange(num_elem):
        sens_factor = penal * (xPhys[i] ** (penal - 1)) * (Emax - Emin)
        elem_dobj = 0.0
        for r in range(8):
            adj_r = adj[edofMat[i, r], 0]
            for c in range(8):
                elem_dobj += adj_r * KE[r, c] * u[edofMat[i, c], 0]
        dobj[i] = sens_factor * elem_dobj


# ----------------------------------------------------------------------------
# Topology optimisation problem
# ----------------------------------------------------------------------------
class TopOptProblem:
    def __init__(self, nelx, nely, volfrac, penal, rmin, ft, plot_history):
        self.nelx         = nelx
        self.nely         = nely
        self.volfrac      = volfrac
        self.penal        = penal
        self.ft           = ft
        self.plot_history = plot_history
        self.Emin  = 1e-9
        self.Emax  = 100.0

        # Spring stiffness — matches reference exactly
        self.kspring = 2
        self.dof_in  = 0

        # ---- passive elements ----
        hex_nut_config = get_hex_nut_elements(nelx, nely, sym_x=True)
        self.solid = hex_nut_config['solid']
        self.void  = hex_nut_config['void']

        # ---- output DOFs: union of top and bottom contact nodes ----
        self.dof_out_dist = np.union1d(
            np.array(hex_nut_config['load_top'],    dtype=np.int32),
            np.array(hex_nut_config['load_bottom'], dtype=np.int32),
        ).astype(np.int32)
        self.num_out_nodes = len(self.dof_out_dist)

        self.ndof = 2 * (nelx + 1) * (nely + 1)

        # ---- design variables ----
        self.x    = volfrac * np.ones(nely * nelx, dtype=np.float64)
        self.xmin = np.zeros(nely * nelx, dtype=np.float64)
        self.xmax = np.ones(nely * nelx,  dtype=np.float64)

        self.xmin[self.solid] = 0.999;  self.xmax[self.solid] = 1.0;   self.x[self.solid] = 1.0
        self.xmin[self.void]  = 0.0;    self.xmax[self.void]  = 0.001; self.x[self.void]  = 0.0
        self.xPhys = self.x.copy()

        # ---- element stiffness matrix ----
        self.KE = lk()

        # ---- DOF connectivity ----
        ely, elx = np.meshgrid(np.arange(nely), np.arange(nelx))
        ely, elx = ely.flatten(), elx.flatten()
        n1 = (nely + 1) * elx + ely
        n2 = (nely + 1) * (elx + 1) + ely
        self.edofMat = np.column_stack(
            [2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 2*n2, 2*n2+1, 2*n1, 2*n1+1]
        ).astype(np.int32)

        self.iK_base = np.kron(self.edofMat, np.ones((8, 1))).flatten().astype(np.int32)
        self.jK_base = np.kron(self.edofMat, np.ones((1, 8))).flatten().astype(np.int32)

        # ---- density filter ----
        import scipy.spatial as spatial
        y_coord, x_coord = np.meshgrid(np.arange(nely), np.arange(nelx))
        coords      = np.column_stack((x_coord.flatten(), y_coord.flatten()))
        tree        = spatial.cKDTree(coords)
        sparse_dist = tree.sparse_distance_matrix(tree, rmin, output_type='coo_matrix')
        iH = sparse_dist.row.astype(np.int32)
        jH = sparse_dist.col.astype(np.int32)
        sH = np.maximum(0.0, rmin - sparse_dist.data)
        self.H  = coo_matrix((sH, (iH, jH)), shape=(nelx*nely, nelx*nely)).tocsc()
        self.Hs = self.H.sum(1)

        # ---- boundary conditions — exact replica of topopt_gripper.py ----
        dofs     = np.arange(self.ndof, dtype=np.int32)
        sym_dofs = dofs[1 : self.ndof : 2*(nely+1)]
        anchor   = np.array([2*(nely+1)-2, 2*(nely+1)-1], dtype=np.int32)
        fixed    = np.union1d(sym_dofs, anchor)
        self.free = np.setdiff1d(dofs, fixed)

        # ---- load and adjoint RHS ----
        self.f      = np.zeros((self.ndof, 1), dtype=np.float64)
        self.u      = np.zeros((self.ndof, 1), dtype=np.float64)
        self.adj    = np.zeros((self.ndof, 1), dtype=np.float64)
        self.f[self.dof_in, 0] = 1.0

        self.dobjdu = np.zeros((self.ndof, 1), dtype=np.float64)
        self.dobjdu[self.dof_out_dist, 0] = 1.0 / self.num_out_nodes

        # ---- spring triplets ----
        self.iK_springs = np.concatenate(([self.dof_in], self.dof_out_dist)).astype(np.int32)
        self.jK_springs = self.iK_springs.copy()
        self.sK_springs = np.concatenate((
            [self.kspring],
            np.full(self.num_out_nodes, self.kspring / self.num_out_nodes, dtype=np.float64)
        ))

        self.iK = np.concatenate((self.iK_base, self.iK_springs))
        self.jK = np.concatenate((self.jK_base, self.jK_springs))

        # ---- working arrays ----
        self.dv                 = np.ones(nely * nelx, dtype=np.float64)
        self.dobj               = np.ones(nely * nelx, dtype=np.float64)
        self.sK_elements_buffer = np.zeros(nelx * nely * 64, dtype=np.float64)

        # ---- iteration bookkeeping ----
        self.loop        = 0
        self.change      = 1.0
        self.last_x      = None
        self.cached_obj  = 0.0
        self.cached_dobj = np.zeros(nelx * nely)
        self.cached_vol  = 0.0
        self.cached_dvol = np.zeros(nelx * nely)
        self.cached_volfrac  = volfrac
        self.t_last_eval_end = time.perf_counter()

        # ---- colours ----
        self.rgb_mint = np.array([62, 180, 137]) / 255.0
        self.rgb_pink = np.array([255, 192, 203]) / 255.0

        if self.plot_history:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(self.generate_rgb_mesh(), interpolation='none')
            self.fig.show()

        print(f"{'It':<4} | {'u_out':<12} | {'Vol':<5} | {'Chg':<5} || ms")
        print("-" * 58)

    # -----------------------------------------------------------------------
    def generate_rgb_mesh(self):
        threshold = 0.1
        is_solid = (self.xPhys > threshold).astype(float)
        base_gray = (1.0 - is_solid).flatten()
        rgb_flat  = np.column_stack([base_gray, base_gray, base_gray])
        rgb_flat[self.solid] = self.rgb_mint
        rgb_flat[self.void]  = self.rgb_pink
        rgb_mesh  = rgb_flat.reshape((self.nelx, self.nely, 3))
        full_mesh = np.hstack([np.fliplr(rgb_mesh), rgb_mesh])
        return np.transpose(full_mesh, (1, 0, 2))

    # -----------------------------------------------------------------------
    def evaluate(self, x_current):
        if self.last_x is not None and np.array_equal(x_current, self.last_x):
            return

        t_oh   = (time.perf_counter() - self.t_last_eval_end) * 1000
        xold1  = self.last_x if self.last_x is not None else x_current
        self.change = np.linalg.norm(x_current - xold1, np.inf)
        self.loop  += 1

        # --- 1. Filter & Heaviside Projection (Sigmund / Wang) ---
        if self.ft == 0:
            xPhys_f = x_current.copy()
            dxPhys_dxtilde = np.ones_like(x_current)
        else:
            # Standard density filter
            x_tilde = np.asarray(self.H * x_current[np.newaxis].T / self.Hs)[:, 0]
            
            # Robust Heaviside Projection
            beta = 8.0  # Sharpness parameter (forces crisp black/white structures)
            eta = 0.5   # Threshold 
            
            denom = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
            xPhys_f = (np.tanh(beta * eta) + np.tanh(beta * (x_tilde - eta))) / denom
            
            # Chain rule derivative of the projection for sensitivities
            dxPhys_dxtilde = beta * (1.0 - np.tanh(beta * (x_tilde - eta))**2) / denom

        self.xPhys = xPhys_f.copy()
        self.xPhys[self.solid] = 1.0
        self.xPhys[self.void]  = 0.0
        
        # Zero out derivatives in passive regions
        dxPhys_dxtilde[self.solid] = 0.0
        dxPhys_dxtilde[self.void] = 0.0

        # --- 2. Assemble ---
        t0 = time.perf_counter()
        fast_assembly(self.xPhys, self.penal, self.Emin, self.Emax,
                      self.KE, self.sK_elements_buffer)
        sK     = np.concatenate((self.sK_elements_buffer, self.sK_springs))
        K      = coo_matrix((sK, (self.iK, self.jK)),
                             shape=(self.ndof, self.ndof)).tocsc()
        K_free = K[self.free, :][:, self.free]
        t_asm  = (time.perf_counter() - t0) * 1000

        # --- 3. Solve ---
        t0 = time.perf_counter()
        
        # FIX: Removed the minus sign on dobjdu to calculate the adjoint for maximizing displacement
        if HAS_PARDISO:
            self.u[self.free, 0]   = pardiso_solve(K_free,  self.f[self.free, 0])
            self.adj[self.free, 0] = pardiso_solve(K_free,  self.dobjdu[self.free, 0])
        else:
            lu = splu(K_free, permc_spec='MMD_AT_PLUS_A')
            self.u[self.free, 0]   = lu.solve(self.f[self.free, 0])
            self.adj[self.free, 0] = lu.solve(self.dobjdu[self.free, 0])
        t_sol = (time.perf_counter() - t0) * 1000

        # --- 4. Objective + Sensitivities ---
        t0 = time.perf_counter()

        # FIX: Negative sign applied here. Minimizing this maximizes the jaw closure.
        obj = -np.sum(self.u[self.dof_out_dist, 0]) / self.num_out_nodes

        fast_sensitivities(self.xPhys, self.penal, self.Emin, self.Emax,
                           self.u, self.adj, self.edofMat, self.KE, self.dobj)

        self.dv[:]          = 1.0
        self.dv[self.solid] = 0.0
        self.dv[self.void]  = 0.0

        vol_scalar = 1.0 / (self.volfrac * self.nely * self.nelx)

        if self.ft == 0:
            self.cached_dobj[:] = (
                np.asarray((self.H * (x_current * self.dobj))[np.newaxis].T / self.Hs)[:, 0]
                / np.maximum(0.001, x_current)
            )
            self.cached_dvol[:] = self.dv * vol_scalar
        else:
            # Apply the Heaviside chain-rule derivative BEFORE passing backward through the density filter
            dobj_proj = self.dobj * dxPhys_dxtilde
            dvol_proj = self.dv * dxPhys_dxtilde
            
            self.cached_dobj[:] = np.asarray(
                self.H * (dobj_proj[np.newaxis].T / self.Hs)
            )[:, 0]
            self.cached_dvol[:] = np.asarray(
                self.H * (dvol_proj[np.newaxis].T / self.Hs)
            )[:, 0] * vol_scalar

        self.cached_obj     = obj
        self.cached_vol     = np.sum(self.xPhys) / (self.volfrac * self.nely * self.nelx) - 1.0
        self.cached_volfrac = np.sum(self.xPhys) / (self.nely * self.nelx)

        t_sen   = (time.perf_counter() - t0) * 1000
        t_total = t_asm + t_sol + t_sen + t_oh

        if self.plot_history:
            self.im.set_array(self.generate_rgb_mesh())
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

        print(f"{self.loop:<4} | {obj:<12.5f} | {self.cached_volfrac:<5.3f} "
              f"| {self.change:<5.3f} || {t_total:.1f}")

        self.last_x          = x_current.copy()
        self.t_last_eval_end = time.perf_counter()

# ----------------------------------------------------------------------------
# Element stiffness matrix
# ----------------------------------------------------------------------------
def lk():
    E  = 1.0
    nu = 0.3
    k  = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
                   -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    KE = E / (1 - nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
    ])
    return KE


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(nelx, nely, volfrac, penal, rmin, ft, plot_history=False, SAVE_FIGURE=True):
    print("Gripper topology optimisation — Original Svanberg MMA")
    print(f"ndes: {nelx} x {nely}   volfrac: {volfrac}   rmin: {rmin}   penal: {penal}")
    print(f"Filter: {'sensitivity' if ft==0 else 'density'}   Live plot: {plot_history}")
    print("-" * 60)

    problem = TopOptProblem(nelx, nely, volfrac, penal, rmin, ft, plot_history)

    # ---- Initialize original MMA parameters & asymptotes ----
    x = problem.x.copy()
    xold1 = x.copy()
    xold2 = x.copy()
    low = np.zeros((nely * nelx, 1))
    upp = np.ones((nely * nelx, 1))

    m_mma = 1  # Number of constraints (volume)
    a0mma = 1.0
    amma  = np.zeros((m_mma, 1))
    cmma  = 1000.0 * np.ones((m_mma, 1))
    dmma  = np.zeros((m_mma, 1))

    loop = 0
    change = 1.0

    # ---- Manual Svanberg MMA Optimization Loop ----
    while change > 0.01 and loop < 600:
        loop += 1
        
        # 1. Execute physical system evaluations via the class
        problem.evaluate(x)
        
        # 2. Reshape 1D optimization outputs to fit mmasub's required 2D shapes
        f0val = problem.cached_obj
        df0dx = problem.cached_dobj[:, np.newaxis]
        fval  = np.array([[problem.cached_vol]])
        dfdx  = problem.cached_dvol[:, np.newaxis].T 
        
        # 3. Compute next step bounds and values using pristine Svanberg routine
        (xnew, _, _, _, _, _, _, _, _, low, upp) = mmasub(
            m_mma, nely * nelx, loop,
            x[:, np.newaxis], problem.xmin[:, np.newaxis], problem.xmax[:, np.newaxis],
            xold1[:, np.newaxis], xold2[:, np.newaxis],
            f0val, df0dx,
            fval, dfdx,
            low, upp, a0mma, amma, cmma, dmma
        )

        # 4. Save structural records and extract step modification
        xold2[:] = xold1
        xold1[:] = x
        x[:] = xnew[:, 0]
        
        change = np.linalg.norm(x - xold1, np.inf)

    # Force a final evaluation to synchronize plots and variables with final design variables
    problem.evaluate(x)

    print("-" * 60)
    print("Optimisation finished.")

    if plot_history:
        plt.ioff()

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Optimised Gripper Design — Hex Nut", fontsize=16, fontweight='bold')
    ax.set_title(
        f"nelx={problem.nelx}  nely={problem.nely*2} (reflected)  "
        f"volfrac={problem.volfrac}  rmin={rmin}  penal={problem.penal}",
        fontsize=11, color='#444',
    )
    ax.imshow(problem.generate_rgb_mesh(), interpolation='none')
    ax.set_xticks(np.arange(0, problem.nelx   + 1, 10))
    ax.set_yticks(np.arange(0, 2*problem.nely  + 1, 10))

    stats = (f"Final u_out: {problem.cached_obj:.5f}  |  "
             f"Vol frac: {problem.cached_volfrac:.3f}  |  "
             f"Iters: {problem.loop}")
    fig.text(0.5, 0.04, stats, ha='center', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                       edgecolor='#ccc', alpha=0.9))
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if SAVE_FIGURE:
        plt.savefig("Images/Hex_gripper_design_.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Create a binary mask of your design
    binary_design = (problem.xPhys.reshape((problem.nelx, problem.nely)) > 0.1).astype(np.uint8)

    # Reflect the design to create the full gripper (as you do in generate_rgb_mesh)
    full_design = np.hstack([np.fliplr(binary_design), binary_design])

    # Save as a standard file
    np.save("final_gripper.npy", full_design)


if __name__ == "__main__":
    # Settings match original topopt_gripper execution values
    main(nelx=240, nely=60, volfrac=0.3, rmin=2.2, penal=3.0,
         ft=1, plot_history=False, SAVE_FIGURE=True)