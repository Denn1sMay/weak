import sympy
from .scripts.integral.util.operators.operators import div, grad, curl
from .scripts.integral.util.boundaries.boundaries import Boundaries
from .scripts.integral.integral import Integral
from .weak_form import Weak_form
from typing import Optional

'''
from dolfinx import mesh, io, plot
from dolfinx.fem import (Constant, Function, FunctionSpace, VectorFunctionSpace,
                         assemble_scalar, dirichletbc, form, locate_dofs_geometrical)
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import grad, div, curl, inner, ds, dx, TrialFunction, TestFunction
from ufl.core.expr import Expr
mesh = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
Q = FunctionSpace(mesh, ("CG", 1))  # Funktionenraum für die Geschwindigkeit
u = TrialFunction(Q)
v = TestFunction(Q)
f = Constant(mesh, ScalarType(-6))


weak_form_object = Weak_form(trial_function_names=["u"], test_function_names=["v"], string_equation="Laplacian(u) = f") #, boundary_condition=Boundaries.neumann, boundary_function="g")

a_generated_string, L_generated_string = weak_form_object.solve()

a_as_dolfin_expr = eval(a_generated_string)
L_as_dolfin_expr = eval(L_generated_string)
'''

weak_form_object = Weak_form(trial_function_names=["u"], test_function_names=["v"], string_equation="Laplacian(u) = f") #, boundary_condition=Boundaries.neumann, boundary_function="g")

a_generated_string, L_generated_string = weak_form_object.solve()
