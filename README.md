# maths

Mathematics self-study with progressive leveling.

**Languages:** Python, LEAN, maybe Julia

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Check your progress
python progress.py
python progress.py --level 0 --verbose

# Run tests to validate solutions
pytest coursework/0-foundations-of-mathematics/ -v
pytest -m level0  # run all Level 0 tests
```

## How It Works

Each level contains problem files with stub functions (`raise NotImplementedError`).
Implement the functions, then run the tests to verify your solutions. Complete all
problems in a level to "unlock" the next one.

## Coursework

### Level 0: Foundations of Mathematics
- Logic and Proofs
- Sets and Number Theory
- Basic Algebra
- Introduction to Functions

### Level 1: Discrete Mathematics and Combinatorics
- Combinatorics and Counting
- Graph Theory
- Discrete Structures (Boolean algebra, lattices)

### Level 2: Calculus and Analysis
- Multivariable Calculus
- Differential Equations
- Partial Differential Equations (PDEs)

### Level 3: Optimization
- Linear Programming
- Nonlinear Programming
- Gradient Descent

### Level 4: Probability and Statistics
- Probability and Stochastic Processes
- Statistical Modeling and Inference
- Data Analysis and Big Data Analytics

### Level 5: Numerical and Scientific Computing
- Numerical Analysis (root finding, interpolation, integration, linear algebra, ODE solvers)
- Scientific Computing and HPC (sparse matrices, FFT, spectral methods)
- Monte Carlo Methods and Stochastic Simulation

### Level 6: Machine Learning and Data Science
- Statistical Learning (classification, dimensionality reduction, clustering)
- Neural Networks (perceptrons, backprop, automatic differentiation)
- Bayesian Methods (Gaussian processes, variational inference)

### Level 7: Advanced Simulation and Data-Integrated Methods
- Uncertainty Quantification (polynomial chaos, sensitivity analysis)
- Inverse Problems (parameter estimation, deblurring, regularization)
- Data-Integrated Methods (physics-informed NNs, Bayesian optimization)

### Level 8: Applied Domains
- Fluid Dynamics Simulation
- Computational Biology
- Energy Systems Modeling

## Project Structure

```
coursework/
  0-foundations-of-mathematics/
    logic-and-proofs/
      problems/       # Problem stubs to implement
      resources/      # Reference materials
      tests/          # Pytest validation
    sets-and-number-theory/
    basic-algebra/
    introduction-to-functions/
  1-discrete-mathematics-and-combinatorics/
  ...
  8-applied-domains/
progress.py           # Check completion status
pyproject.toml        # Dependencies and config
conftest.py           # Test infrastructure
```
