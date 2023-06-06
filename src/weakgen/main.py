from scripts.integral.integral import Integral
import sympy
from scripts.integral.util.dimensions.dimensions import Dimensions
from scripts.integral.util.operators.operators import div, grad, rot
from scripts.integral.util.boundaries.boundaries import Boundaries
from weak_form import Weak_form
from typing import Optional


u = sympy.Symbol("u")
v = sympy.Symbol("v")
ex = sympy.Expr(grad(u))

integral = Integral(ex, u, v)



weak_form_object = Weak_form(equation = None, trial_function_name="u", test_function_name="v", vector_trial_fuction_name="u_vec", vector_test_function_name="v_vec", string_equation="m*Laplacian(u) = f") #, boundary_condition=Boundaries.neumann, boundary_function="g")

weak_form_object.multiply_with_test_function()
weak_form_object.integrate_over_domain()
weak_form_object.integraty_by_parts()
weak_form_object.convert_to_ufl_string()
print(weak_form_object.lhs_ufl_string)
print(weak_form_object.rhs_ufl_string)
