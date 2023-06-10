import sympy
from .scripts.integral.util.operators.operators import div, grad, curl
from .scripts.integral.util.boundaries.boundaries import Boundaries
from .scripts.integral.integral import Integral
from .weak_form import Weak_form
from typing import Optional



weak_form_object = Weak_form(sympy_equation = None,  vector_trial_fuction_names=["u_vec"], trial_function_names=["u", "q"], test_function_names=["v", "w"], vector_test_function_names=["v_vec"], string_equation="Laplacian(q) = f") #, boundary_condition=Boundaries.neumann, boundary_function="g")

weak_form_object.solve()
