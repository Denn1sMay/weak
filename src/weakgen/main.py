import sympy
from .scripts.integral.util.operators.operators import div, grad, curl
from .scripts.integral.util.boundaries.boundaries import Boundaries
from .scripts.integral.integral import Integral
from .weak_form import Weak_form
from typing import Optional


u = sympy.Symbol("u")
v = sympy.Symbol("v")
ex = sympy.Expr(grad(u))

integral = Integral(ex, u, v)



weak_form_object = Weak_form(equation = None, trial_function_name="u", test_function_name="v", vector_trial_fuction_name="u_vec", vector_test_function_name="v_vec", string_equation="div(u_vec)") #, boundary_condition=Boundaries.neumann, boundary_function="g")

weak_form_object.solve()
