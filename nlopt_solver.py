import nlopt

def run_nlopt_mma(x0, xmin, xmax, obj_func, constraint_func, max_eval=200, xtol=0.01, vol_tol=1e-6):
    """
    Wrapper for NLopt's Method of Moving Asymptotes (MMA).
    
    This function initializes the optimizer, sets the physical limits of the 
    design variables, applies the volume constraint, and strictly manages the 
    termination criteria to prevent premature convergence.
    
    Args:
        x0 (np.ndarray): Initial design variable array (densities).
        xmin (np.ndarray): Minimum allowable bounds for each element.
        xmax (np.ndarray): Maximum allowable bounds for each element.
        obj_func (callable): Function returning the objective value and updating its gradient.
        constraint_func (callable): Function returning the constraint value and updating its gradient.
        max_eval (int): Maximum number of iterations before forcing a stop.
        xtol (float): Absolute tolerance for topology change between iterations.
        vol_tol (float): Tolerance for the volume inequality constraint.
    """
    # Initialize the MMA optimizer with the number of design variables
    opt = nlopt.opt(nlopt.LD_MMA, len(x0))
    
    # Set the physical boundaries (typically 0 to 1, with exceptions for solid/void regions)
    opt.set_lower_bounds(xmin)
    opt.set_upper_bounds(xmax)
    
    # Assign the core evaluation functions
    opt.set_min_objective(obj_func)
    opt.add_inequality_constraint(constraint_func, vol_tol)
    
    # Disable native relative tolerances. 
    # TopOpt gradients can flatten out, which falsely triggers relative convergence in NLopt.
    opt.set_ftol_rel(0.0)
    opt.set_xtol_rel(0.0)
    opt.set_ftol_abs(0.0)
    
    # Rely strictly on absolute step size and the maximum iteration limit
    opt.set_maxeval(max_eval)
    opt.set_xtol_abs(xtol)
    
    try:
        # Hand control over to the C-backend to run the optimization loop
        opt.optimize(x0)
    except nlopt.RoundoffLimited:
        # Expected in Topology Optimization when the mesh completely converges 
        # and numerical precision limits further mathematical progress.
        pass 
    except nlopt.ForcedStop:
        # Triggered manually by our physics engine when the custom criteria are met.
        pass
    except Exception as e:
        # Catch and print any other C-level exceptions from NLopt
        print(f"NLopt terminated: {e}")