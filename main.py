from src.integral.integral import Integral
import sympy
from src.integral.util.dimensions.dimensions import Dimensions
from src.integral.util.operators.operators import div, grad, rot
from src.integral.util.boundaries.boundaries import Boundaries
from src.weak_form import Weak_form
from typing import Optional


u = sympy.Symbol("u")
v = sympy.Symbol("v")
ex = sympy.Expr(grad(u))

integral = Integral(ex, u, v)



xxx = Weak_form(equation = None, trial_function_name="u", test_function_name="v", vector_trial_fuction_name="u_vec", vector_test_function_name="v_vec", string_equation="Laplacian(u) = f", boundary_condition=Boundaries.neumann, boundary_function="g")

xxx.multiply_with_test_function()
xxx.integrate_over_domain()
xxx.integraty_by_parts()
print(xxx.equation)
'''
for term in xxx.lhs_terms:
    print("Term:")
    print(term.term)
'''
