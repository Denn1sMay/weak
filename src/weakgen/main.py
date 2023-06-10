import sympy
from .scripts.integral.util.operators.operators import div, grad, curl
from .scripts.integral.util.boundaries.boundaries import Boundaries
from .scripts.integral.integral import Integral
from .weak_form import Weak_form
from typing import Optional



weak_form_object = Weak_form(sympy_equation = None, trial_function_names=["u", "q"], test_function_names=["v", "w"], string_equation="u = f") #, boundary_condition=Boundaries.neumann, boundary_function="g")

weak_form_object.solve()
