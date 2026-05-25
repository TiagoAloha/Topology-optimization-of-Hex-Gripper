# Topology Optimization of a Compliant Hex Gripper

This repository contains the Finite Element Method (FEM) codebase and non-linear solvers for the topology optimization of a compliant gripper. Designed specifically for handling hexagonal geometries like standard nuts and bolts, the optimization process is structured to yield functional mechanisms ready for additive manufacturing (AM) production.

This project was developed as part of the Additive Manufacturing design coursework within the Aerospace Structures and Materials (ASM) program at TU Delft.

## Overview

The core objective of this codebase is to iteratively design a structural layout that maximizes targeted mechanical compliance while minimizing overall material volume. The architecture leverages Python-based environments to evaluate and optimize the design, preparing the final geometry for additive manufacturing techniques. 

A critical structural constraint explicitly enforced within the finite element implementation is the **symmetry condition at the `y=0` plane**. This ensures the resulting geometry maintains a perfectly balanced gripping actuation while significantly reducing the computational overhead during solver iterations.

## Repository Structure

The project is structured with modularity in mind, cleanly separating the problem domain, the FEM calculations, and the mathematical solvers.

* **`topopt_gripper.py` (and variants)**: The main executable scripts for running the topology optimization routines. Different variations (`_opt.py`, `_opt_2.py`, etc.) evaluate the gripper against varying conditions or distinct solver methods.
* **`domain_config.py`**: Handles the boundary conditions, input loads, and initial mesh definitions for the structural domain, including the enforcement of the `y=0` symmetry constraint.
* **`mma.py`**: Implementation of the Method of Moving Asymptotes (MMA), an algorithm highly effective for handling large-scale, constrained non-linear optimization problems typical in topology optimization.
* **`nlopt_solver.py`**: Integration with the NLopt suite to leverage its robust, open-source library of non-linear optimization algorithms.
* **`grid_to_STL.py`**: A vital utility script that converts the optimized density arrays directly into standard `.stl` formats, bridging the gap between computational simulation and AM preparation software.
* **`Assignment_2_AM.pdf`**: The underlying project brief and design parameters.

## Optimization Methodology

The optimization relies on a density-based approach (SIMP method) where the material distribution within the defined structural domain is heavily penalized to achieve a clear solid-versus-void design. 

By utilizing both **MMA** and **NLopt** routines, the code systematically updates the material distribution. The solvers evaluate the sensitivity of the compliance matrix to adjust the structure at each step, carving away inactive regions until the ideal compliant mechanism remains.

## Additive Manufacturing Considerations

Because the mechanism is intended for direct 3D printing, the resulting theoretical output requires careful handling. Producing functional compliance via additive manufacturing—especially in metal—requires addressing toolpath generation constraints and potential anisotropic material behaviors. The provided `grid_to_STL.py` script acts as the first step in this translation, allowing the optimized result to be easily exported for subsequent non-planar slicing or immediate print preparation.

## Getting Started

Ensure you have Python installed along with standard scientific libraries (`numpy`, `scipy`). If you are running the NLopt solver scripts, you will also need the `nlopt` package installed in your environment:

```bash
pip install numpy scipy nlopt