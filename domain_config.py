import numpy as np
from functools import reduce


def get_hex_nut_elements(nelx, nely, sym_x=False):
    """
    Calculates and returns the 1D arrays of passive solid and void elements.
    """
    # --- Hexagonal Nut Passive Region Settings ---
    # Set center to 0 if modeling the right half, otherwise place in the middle
    xc =  int(7 * nelx / 8) 
    yc = 0 if sym_x else int(nely / 2) #symmetrical in y direction (around x-axis)      
    
    R_hex_in = nely / 8 # Inner radius of the hexagon (distance from center to a vertex)
    R_out = R_hex_in + 2 # A small sliver of material around hex to grip the nut               
             

    elx_grid, ely_grid = np.meshgrid(np.arange(nelx), np.arange(nely), indexing='ij')
    
    dx = np.abs(elx_grid + 0.5 - xc)
    dy = np.abs(ely_grid + 0.5 - yc)
    dist = np.sqrt(dx**2 + dy**2)
    
    apothem = R_hex_in * (np.sqrt(3) / 2)
    inside_hex = (dy <= apothem) & ((dx * (np.sqrt(3) / 2) + dy * 0.5) <= apothem)
    
    linear_indices = elx_grid * nely + ely_grid

    nut_void = linear_indices[inside_hex].astype(int)
    nut_solid = linear_indices[(dist <= R_out) & ~inside_hex].astype(int)

    # --- NEW: Crescent / Open-Ended Mask ---
    # Defines a channel that cuts through the outer solid ring.
    # The width matches the flat sides of the hexagon (dy <= apothem).
    # Currently set to open towards the RIGHT: elx_grid > xc
    # -> To flip LEFT: change to `elx_grid < xc`
    # -> To give the crescent deeper "arms" that wrap past the center: 
    #    change `xc` to `(xc + R_hex_in * 0.5)`
    is_opening = (elx_grid > xc) & (dy <= apothem)
    
    # Flattened linear index map
    linear_indices = elx_grid * nely + ely_grid

    # The void is exactly the inside of the hexagon
    nut_void = linear_indices[inside_hex]
    
    # The solid is the outer circle, MINUS the hexagon, MINUS the opening channel
    nut_solid = linear_indices[(dist <= R_out) & ~inside_hex & ~is_opening]
    # ---------------------------------------------

    # Define final passive solid domains (manual addition of solids and voids)
    solid = reduce(np.union1d, (
        np.array([x*nely+y for x in range(5) for y in range(3)], dtype=int), # Load point 
        # np.array([x*nely+y for x in range(int(3*nelx/4), nelx) for y in range(int(nely/2), int(nely/2)+3)], dtype=int), 
        # np.array([x*nely+y for x in range(int(3*nelx/4)-5, int(3*nelx/4)) for y in range(3)], dtype=int),
        nut_solid 
    ))
    
    void = reduce(np.union1d, (
        np.array([], dtype=int),
        nut_void 
    ))
    
    # --- DISTRIBUTED LOAD DOFS CALCULATION ---
    # Identify the horizontal node span over the nut area
    x_nodes = np.arange(xc, xc + int(R_out / 2) + 1)
    
    # Bottom load calculation (Always present in the bottom-half domain)
    y_bottom_node = min(nely, yc + R_out)
    bottom_nodes = (nely + 1) * x_nodes + y_bottom_node
    dofs_bottom_y = 2 * bottom_nodes.astype(int) + 1
    
    # Top load calculation (Omitted if sym_x cuts out the top half because it is above y=0)
    if sym_x:
        dofs_top_y = np.array([], dtype=int)
    else:
        y_top_node = max(0, yc - R_out)
        top_nodes = (nely + 1) * x_nodes + y_top_node
        dofs_top_y = 2 * top_nodes.astype(int) + 1
    
    return dict({
        'solid': solid,
        'void': void,
        'load_top': dofs_top_y,
        'load_bottom': dofs_bottom_y
    })