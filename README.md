# Weak Form Equation Converter

The Weak Form Equation Converter is a Python program designed to automate the generation of weak-form equations for finite element analysis. It provides a convenient way to convert a strong-form equation into a weak form that can be used in numerical simulations. The package uses the ufl-syntax for operators and integrals. It is build on the dolfinx library and can additionally automate the declaration of FiniteElements and FunctionSpaces.

To use the package, install it via pip:
```bash
pip install -i https://test.pypi.org/simple/ weakgen
```
You also need sympy installed for the package to work correctly:
```bash
pip instal sympy
```
___
## Example Usage

```python
from mpi4py import MPI
from dolfinx import mesh, fem

from weakgen import Weak_form, Boundaries

domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)

u_dict = {
    "u": {
        "order": 1,
        "dim": "scalar",
        "spaceName": "Myspace"
    }
}

boundaryFunctions = {"curl": "g_curl", "grad": "g_grad", "div": "g_div", "laplacian": "g_lap"}

f = fem.Constant(domain, ScalarType(-6))
pde = "Laplacian(u) = f"

weak_form_object = Weak_form(functions=u_dict, mesh="domain", string_equation=pde, boundary_condition=Boundaries.neumann, boundary_function=boundaryFunctions)

# lhs and rhs of equation as strings
# commands contains a python-executable script
a_generated_string, L_generated_string, commands = weak_form_object.solve()
# Declares Finite Elements and FunctionSpaces, TrialFunctions and Testfunctions
# Declares lhs of the Equation as "a" and rhs as "L"
exec(commands)

# After defining your Boundary conditions "bc"
# The variables a and L can be used like that:

problem = fem.petsc.LinearProblem(a, L, bcs=[bc])
uh = problem.solve()
```

___

### Input
To use the package, provide the strong form of the equation as a string to the `equation_string` parameter. The equation can be scalar- or vector-valued, as it will be converted to a scalar weak form.
You also need to provide the unknown functions of your equation. Define them as a dict like:
```python
{dict_name} = {
    "{variable_name}": {
        "order": {number},
        "dim": {"scalar" | "vector" | "matrix"},
        "spaceName": {optional: string}
    }
}

# Example:
variables = {
    "u": {
        "order": 2,
        "dim": "vector",
    },
    "p": {
        "order": 1,
        "dim": "scalar",
        "spaceName": "my_functionspace_name"
    }
}

```
You have to provide the polynomial order to the "order"-key and the dimension of your function to the "dim"-key. The "dim" "matrix" corresponds to a helper dimension. It is assumed that you apply a function to a defined TrialFunction, which results in a matrix. The matrix-varialbe will not be initializes inside the "commands"-script!
Optionally, you can pass the desired name of the Functionspace for each variable. 

Pass the variable name of your defined mesh to the "mesh"-parameter.

All Functions and Functionspaces will be initialized by calling the "exec()"-Function with the "commands"-variable, which is returned by the "solve"- method of your Weak_form-object.

If you want to initilize all functions and spaces on your own, you can use pythons "eval()"-function on the received "a_generated_string" and "L_generated_string"-variables.

The name of the (mixed) function space defaults to "V", if not specified otherwise. Sub function spaces will be accessible with "V_1", "V_2",... if not specified otherwise.
The testfunctions will be named like the unknown function with the postfix "_test". The example would produce "u_test" and "p_test".

If you encounter any difficulties during the conversion process, you can set the `debug=True` parameter to receive hints on where the conversion failed.
___
### Accessible differential operators

The package supports the following differential operators:

- Laplacian (take the laplacian of an expression)
- grad (take the gradient of an expression)
- div (take the divergence of an expression)
- curl (take the curl of an exression)

### Applicable Boundary Conditions
You can specify a boundary condition by passing a condition to the `boundary_condition` parameter. If you need a Neumann boundary condition, you must provide a string literal for the `boundary_function` parameter. The available options are:
- Boundaries.dirichlet (default): Surface integrals will vanish.
- Boundaries.neumann: Requires the `boundary_function` parameter. Applies a Neumann boundary condition.

To specify boundary functions, also use a dict with following keys - only specify required keys:
```python
boundaryFunctions = {"curl": "g_curl", "grad": "g_grad", "div": "g_div", "laplacian": "g_lap"}
```
g_curl corresponds to (n x u), g_div to (n ⋅ u), etc.
___

### Troubleshooting
If you encounter errors, try to pass the names of constant variables to the "variables" or "variable_vectors"-parameter, respectively. This will help to determine the dimension of single expressions.

___

### What does it do
The Weak Form Equation Generator attempts to parse your string equation into a sympy equation and separate its terms. When you call solve() on the returned object, the following steps will be executed:

1. Multiply the equation with the test function.
2. Integrate the equation over the domain.
3. Apply integration by parts to terms containing differential operators.
4. Convert the equation to a UFL-syntaxed string.

The two resulting string values represent the left-hand side (LHS) and the right-hand side (RHS) of the weak form equation. You can use the built-in eval() function in Python to parse the string equation into the variables present in your program's scope.



[GitHub](https://github.com/Denn1sMay/weak)
