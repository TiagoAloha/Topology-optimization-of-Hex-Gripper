from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from matplotlib import colors
import matplotlib.pyplot as plt
from domain_config import get_hex_nut_elements
from numba import njit, prange
import time

from nlopt_solver import run_nlopt_mma

# ----------------------------------------------------------------------------
# Solver Initialization
# Attempt to load PyPardiso for high-performance Symmetric Positive Definite solving.
# Falls back to SciPy's SuperLU if the Intel MKL library is not installed.
# ----------------------------------------------------------------------------
try:
    from pypardiso import spsolve as pardiso_solve
    HAS_PARDISO = True
except ImportError:
    HAS_PARDISO = False


# ----------------------------------------------------------------------------
# JIT-Compiled Physics Loops
# These functions bypass standard Python overhead, mapping mathematical 
# operations directly to CPU cores (prange) for C-like execution speeds.
# ----------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def fast_assembly(xPhys, penal, Emin, Emax, KE, sK_elements):
    """
    Constructs the global stiffness matrix entries.
    Iterates through every element, applies SIMP penalization, and maps the 
    8x8 local element matrix into a flat 1D array for sparse COO matrix creation.
    """
    num_elem = xPhys.shape[0]
    for i in prange(num_elem):
        # Apply SIMP material penalization
        density_penalized = xPhys[i] ** penal
        E_val = Emin + density_penalized * (Emax - Emin)
        
        # Map 8x8 element stiffness to the flat 1D array
        for r in range(8):
            for c in range(8):
                idx = i * 64 + r * 8 + c
                sK_elements[idx] = KE[r, c] * E_val

@njit(cache=True, parallel=True)
def fast_sensitivities(xPhys, penal, Emin, Emax, u, adj, edofMat, KE, dobj):
    """
    Calculates the objective function gradients (sensitivities).
    Uses the displacement vectors (u) and adjoint vectors (adj) solved during 
    the FEM phase to compute how a change in density affects the objective.
    """
    num_elem = xPhys.shape[0]
    for i in prange(num_elem):
        # Calculate the derivative of the SIMP penalization rule
        sens_factor = penal * (xPhys[i] ** (penal - 1)) * (Emax - Emin)
        elem_dobj = 0.0
        
        # Matrix multiplication for the element sensitivity: adj^T * KE * u
        for r in range(8):
            dof_r = edofMat[i, r]
            adj_r = adj[dof_r, 0]
            for c in range(8):
                dof_c = edofMat[i, c]
                u_c = u[dof_c, 0]
                elem_dobj += adj_r * KE[r, c] * u_c
                
        dobj[i] = sens_factor * elem_dobj


# ----------------------------------------------------------------------------
# Physics Engine Class
# ----------------------------------------------------------------------------
class TopOptProblem:
    """
    Object-oriented wrapper that manages the state of the finite element model.
    It caches evaluations to prevent NLopt from triggering duplicate FEM solves 
    for the same design iteration.
    """
    def __init__(self, nelx, nely, volfrac, penal, rmin, ft, plot_history):
        # Core configuration parameters
        self.nelx = nelx
        self.nely = nely
        self.volfrac = volfrac
        self.penal = penal
        self.ft = ft
        self.plot_history = plot_history
        self.Emin = 1e-9
        self.Emax = 100.0
        self.kspring = 1.0
        self.dof_in = 0

        # Load domain specifics (passive solid/void regions and external nodes)
        hex_nut_config = get_hex_nut_elements(nelx, nely, sym_x=True)
        self.solid = hex_nut_config['solid']
        self.void  = hex_nut_config['void']
        self.dof_out_dist = np.array(hex_nut_config['load_bottom'], dtype=np.int32)
        self.num_out_nodes = len(self.dof_out_dist)
        self.ndof = 2 * (nelx + 1) * (nely + 1)

        # Initialize design variables (densities) and bounds
        self.x = volfrac * np.ones(nely * nelx, dtype=np.float64)
        self.xmin = np.zeros(nely * nelx, dtype=np.float64)
        self.xmax = np.ones(nely * nelx, dtype=np.float64)
        
        # Enforce strict boundary conditions on passive regions
        self.xmin[self.solid] = 0.999; self.xmax[self.solid] = 1.0; self.x[self.solid] = 1.0
        self.xmin[self.void] = 0.0; self.xmax[self.void] = 0.001; self.x[self.void] = 0.0
        self.xPhys = self.x.copy()

        # Generate base element stiffness matrix
        self.KE = lk()

        # Generate mesh connectivity matrices
        ely, elx = np.meshgrid(np.arange(nely), np.arange(nelx))
        ely, elx = ely.flatten(), elx.flatten()
        n1 = (nely + 1) * elx + ely
        n2 = (nely + 1) * (elx + 1) + ely
        self.edofMat = np.column_stack([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 2*n2, 2*n2+1, 2*n1, 2*n1+1]).astype(np.int32)

        # Pre-allocate base row/column indices for the sparse stiffness matrix
        self.iK_base = np.kron(self.edofMat, np.ones((8, 1))).flatten().astype(np.int32)
        self.jK_base = np.kron(self.edofMat, np.ones((1, 8))).flatten().astype(np.int32)

        # Mesh-independent filtering setup using spatial trees
        import scipy.spatial as spatial
        y_coord, x_coord = np.meshgrid(np.arange(nely), np.arange(nelx))
        coords = np.column_stack((x_coord.flatten(), y_coord.flatten()))
        tree = spatial.cKDTree(coords)
        sparse_dist = tree.sparse_distance_matrix(tree, rmin, output_type='coo_matrix')
        iH = sparse_dist.row.astype(np.int32)
        jH = sparse_dist.col.astype(np.int32)
        sH = np.maximum(0.0, rmin - sparse_dist.data)
        
        self.H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
        self.Hs = self.H.sum(1)

        # Define degrees of freedom (DOFs) and enforce symmetry boundary conditions
        dofs = np.arange(self.ndof, dtype=np.int32)
        fixed_base = np.array([2*(nely+1)-2, 2*(nely+1)-1], dtype=np.int32)
        symmetry_nodes = np.arange(0, (nelx + 1) * (nely + 1), nely + 1, dtype=np.int32)
        symmetry_y_dofs = 2 * symmetry_nodes + 1
        fixed = np.union1d(fixed_base, symmetry_y_dofs)
        self.free = np.setdiff1d(dofs, fixed)

        # Initialize load and displacement vectors
        self.f = np.zeros((self.ndof, 1), dtype=np.float64)
        self.u = np.zeros((self.ndof, 1), dtype=np.float64)
        self.adj = np.zeros((self.ndof, 1), dtype=np.float64)
        
        # Apply primary load
        self.f[self.dof_in, 0] = 1.0

        # Define the objective node mapping
        self.dobjdu = np.zeros((self.ndof, 1), dtype=np.float64)
        self.dobjdu[self.dof_out_dist, 0] = 1.0 / self.num_out_nodes

        # Setup artificial springs for inverter modeling
        self.iK_springs = np.concatenate(([self.dof_in], self.dof_out_dist)).astype(np.int32)
        self.jK_springs = np.concatenate(([self.dof_in], self.dof_out_dist)).astype(np.int32)
        self.sK_springs = np.concatenate(([self.kspring], np.full(self.num_out_nodes, self.kspring / self.num_out_nodes, dtype=np.float64)))
        
        # Combine element nodes and spring nodes into final sparse matrix indices
        self.iK = np.concatenate((self.iK_base, self.iK_springs))
        self.jK = np.concatenate((self.jK_base, self.jK_springs))

        # Initialize memory buffers for volume derivatives and gradients
        self.dv = np.ones(nely * nelx, dtype=np.float64)
        self.dobj = np.ones(nely * nelx, dtype=np.float64)
        self.sK_elements_buffer = np.zeros(nelx * nely * 64, dtype=np.float64)

        # Caching and Loop variables
        self.loop = 0
        self.change = 1.0
        self.last_x = None
        self.obj_scale = None 
        
        self.cached_obj = 0.0
        self.cached_dobj = np.zeros(nelx * nely)
        self.cached_vol = 0.0
        self.cached_dvol = np.zeros(nelx * nely)
        self.t_last_eval_end = time.perf_counter()

        # Initialize live plotting if enabled
        if self.plot_history:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            xPlot0 = np.round(self.xPhys).reshape((nelx, nely))
            xPlot0 = np.hstack([np.fliplr(xPlot0), xPlot0])
            self.im = self.ax.imshow(-xPlot0.T, cmap='gray', interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
            self.fig.show()

        # Setup table headers for terminal logging
        print(f"{'It':<4} | {'Obj':<8} | {'Vol':<5} | {'Chg':<5} || {'Asm(ms)':<7} | {'Sol(ms)':<7} | {'Sen(ms)':<7} | {'NLopt(ms)':<9} | {'Total(ms)'}")
        print("-" * 94)

    def evaluate(self, x_current):
        """
        The core Physics Sequence. 
        Filters variables -> Assembles matrices -> Solves system -> Computes gradients.
        """
        # Caching: If NLopt asks for the same design twice in a row, return immediately.
        if self.last_x is not None and np.array_equal(x_current, self.last_x):
            return

        t_nlopt_overhead = (time.perf_counter() - self.t_last_eval_end) * 1000
        
        xold1 = self.last_x if self.last_x is not None else x_current
        self.change = np.linalg.norm(x_current - xold1, np.inf)
        self.loop += 1

        # Apply Mesh Filtering (Sensitivity vs Density)
        if self.ft == 0:
            self.xPhys[:] = x_current
        elif self.ft == 1:
            self.xPhys[:] = np.asarray(self.H * x_current[np.newaxis].T / self.Hs)[:, 0]
            
        # Hard-enforce solid and void geometries post-filter
        self.xPhys[self.solid] = 1.0
        self.xPhys[self.void] = 0.0

        # --- Assembly Phase ---
        t0 = time.perf_counter()
        fast_assembly(self.xPhys, self.penal, self.Emin, self.Emax, self.KE, self.sK_elements_buffer)
        sK = np.concatenate((self.sK_elements_buffer, self.sK_springs))
        K = coo_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsc()
        K_free = K[self.free, :][:, self.free]
        t_asm = (time.perf_counter() - t0) * 1000

        # --- Linear Solve Phase ---
        t0 = time.perf_counter()
        if HAS_PARDISO:
            self.u[self.free, 0] = pardiso_solve(K_free, self.f[self.free, 0].flatten())
            self.adj[self.free, 0] = pardiso_solve(K_free, -self.dobjdu[self.free, 0].flatten())
        else:
            # Optimal SciPy SuperLU configuration for 2D Quad element matrices
            solve_K = splu(K_free, permc_spec='MMD_AT_PLUS_A')
            self.u[self.free, 0] = solve_K.solve(self.f[self.free, 0].flatten())
            self.adj[self.free, 0] = solve_K.solve(-self.dobjdu[self.free, 0].flatten())
        t_sol = (time.perf_counter() - t0) * 1000

        # --- Sensitivity Phase ---
        t0 = time.perf_counter()
        self.cached_obj = np.sum(self.u[self.dof_out_dist, 0]) / self.num_out_nodes
        fast_sensitivities(self.xPhys, self.penal, self.Emin, self.Emax, self.u, self.adj, self.edofMat, self.KE, self.dobj)
        self.dv[:] = np.ones(self.nely * self.nelx)

        # Zero out the physical gradients for clamped boundary geometries to avoid 
        # chain-rule mismatches that confuse the optimizer's trust-region logic.
        self.dobj[self.solid] = 0.0
        self.dobj[self.void]  = 0.0
        self.dv[self.solid]   = 0.0
        self.dv[self.void]    = 0.0

        vol_scalar = 1.0 / (self.volfrac * self.nely * self.nelx)

        # Apply spatial filtering to the gradients
        if self.ft == 0:
            self.cached_dobj[:] = np.asarray((self.H * (x_current * self.dobj))[np.newaxis].T / self.Hs)[:, 0] / np.maximum(0.001, x_current)
            self.cached_dvol[:] = self.dv * vol_scalar
        elif self.ft == 1:
            self.cached_dobj[:] = np.asarray(self.H * (self.dobj[np.newaxis].T / self.Hs))[:, 0]
            self.cached_dvol[:] = np.asarray(self.H * (self.dv[np.newaxis].T / self.Hs))[:, 0] * vol_scalar

        # Compute constraint offset (must evaluate <= 0 for NLopt)
        self.cached_vol = np.sum(self.xPhys) / (self.volfrac * self.nely * self.nelx) - 1.0
        t_sen = (time.perf_counter() - t0) * 1000

        t_total = t_asm + t_sol + t_sen + t_nlopt_overhead

        # Handle UI updating
        if self.plot_history:
            xPlot = np.round(self.xPhys).reshape((self.nelx, self.nely))
            xPlot = np.hstack([np.fliplr(xPlot), xPlot])
            self.im.set_array(-xPlot.T)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

        # Log iteration times
        print(f"{self.loop:<4} | {self.cached_obj:<8.5f} | {self.cached_vol+1.0:<5.3f} | {self.change:<5.3f} || "
              f"{t_asm:<7.1f} | {t_sol:<7.1f} | {t_sen:<7.1f} | {t_nlopt_overhead:<9.1f} | {t_total:.1f}")

        # Store iteration state for caching
        self.last_x = x_current.copy()
        self.t_last_eval_end = time.perf_counter()

    def obj_func(self, x, grad):
        """
        NLopt interface function. 
        Triggers the physics evaluation and returns the scaled objective and gradient.
        """
        self.evaluate(x)
        
        # Dynamic scaling initialization (executed on iteration 1).
        # Multiplies tiny gradient magnitudes up to ~10.0 so NLopt is mathematically 
        # encouraged to take large step sizes mimicking Svanberg's original MMA behavior.
        if self.obj_scale is None:
            max_g = np.max(np.abs(self.cached_dobj))
            self.obj_scale = 10.0 / max_g if max_g > 1e-12 else 1.0
            
        # Update gradient array in-place
        if grad.size > 0:
            grad[:] = self.cached_dobj * self.obj_scale
            
        return self.cached_obj * self.obj_scale

    def vol_constraint(self, x, grad):
        """
        NLopt constraint interface function.
        Returns the constraint value and updates the constraint gradient.
        """
        self.evaluate(x)
        
        # Update gradient array in-place
        if grad.size > 0:
            grad[:] = self.cached_dvol
            
        return self.cached_vol


# ----------------------------------------------------------------------------
# Execution Main Block
# ----------------------------------------------------------------------------
def main(nelx, nely, volfrac, penal, rmin, ft, plot_history=False):
    print("Inverter mechanism design with NLopt MMA")
    print("ndes: " + str(nelx) + " x " + str(nely))
    print("volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal))
    print("Live plotting: " + str(plot_history))
    print("-" * 50)

    # Initialize the topology environment and problem state
    problem = TopOptProblem(nelx, nely, volfrac, penal, rmin, ft, plot_history)

    # Execute the external C-based optimizer
    run_nlopt_mma(
        x0=problem.x,
        xmin=problem.xmin,
        xmax=problem.xmax,
        obj_func=problem.obj_func,
        constraint_func=problem.vol_constraint,
        max_eval=200,
        xtol=0.01 
    )

    print("-" * 94)
    print("Optimization finished. Displaying final design.")
    
    # Process final visualization
    if plot_history:
        plt.ioff()
    else:
        fig, ax = plt.subplots()
        xPlot = np.round(problem.xPhys).reshape((nelx, nely))
        xPlot = np.hstack([np.fliplr(xPlot), xPlot])
        im = ax.imshow(-xPlot.T, cmap='gray', interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))

    plt.show()

def lk():
    """
    Constructs the base analytical 8x8 element stiffness matrix for a 2D quad element.
    Assumes a Young's Modulus of 1.0 and a Poisson's ratio of 0.3.
    """
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
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ])
    return KE

if __name__ == "__main__":
    main(nelx=180, nely=60, volfrac=0.5, rmin=4.0, penal=3.0, ft=1, plot_history=False)