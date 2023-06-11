# Weak Form Equation Converter

The Weak Form Equation Converter is a Python program designed to automate the generation of weak-form equations for finite element analysis. It provides a convenient way to convert a strong-form equation into a weak form that can be used in numerical simulations. The package is build on the dolfinx library and uses the corresponding ufl-syntax for operators and integrals.

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
from weakgen import Weak_form
from ufl import inner, grad, div, curl, div, ds, dx

weak_form_object = Weak_form(trial_function_names=["u"], test_function_names=["v"], string_equation="Laplacian(u) = f")
weak_form_lhs_string, weak_form_rhs_string = weak_form_object.solve()
# Result: (-inner(grad(u), grad(v))) * dx = (f*v) * dx
a_as_dolfin_expr = eval(weak_form_lhs_string)
l_as_dolfin_expr = eval(weak_form_rhs_string)
```

To use the package, you need to import the necessary UFL operators (inner, grad, div, curl, ds, dx) as shown in the example. This allows the eval() function to map the string functions to the corresponding UFL implementation.


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

___
### Input
To use the package, provide the strong form of the equation as a string to the `equation_string` parameter. The equation can be scalar- or vector-valued, as it will be converted to a scalar weak form.
You also need to provide the string literals of the trial function(s) in a list to the `trial_function_names` or `vector_trial_function_names` parameter, depending on their dimension. Similarly, provide the string literals of the test function(s) defined in your program's scope to the `test_function_names` or `vector_test_function_names` parameter, based on the dimension of your equation.
If you encounter any difficulties during the conversion process, you can set the `debug=True` parameter to receive hints on where the conversion failed.
___
### What does it do
The Weak Form Equation Generator attempts to parse your string equation into a sympy equation and separate its terms. When you call solve() on the returned object, the following steps will be executed:

1. Multiply the equation with the test function.
2. Integrate the equation over the domain.
3. Apply integration by parts to terms containing differential operators.
4. Convert the equation to a UFL-syntaxed string.

The two resulting string values represent the left-hand side (LHS) and the right-hand side (RHS) of the weak form equation. You can use the built-in eval() function in Python to parse the string equation into the variables present in your program's scope.



[GitHub](https://github.com/Denn1sMay/weak)
