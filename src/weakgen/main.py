import sympy
from .scripts.integral.util.operators.operators import div, grad, curl
from .scripts.integral.util.boundaries.boundaries import Boundaries, BoundaryFunctions
from .scripts.integral.integral import Integral
from .weak_form import Weak_form
from typing import Optional



from mpi4py import MPI
from dolfinx import mesh
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




weak_form_object = Weak_form(trial_function_names=["u", "q"], test_function_names=["v", "w"], vector_test_function_names=["m"], string_equation="inner(grad(u), c) = f") #, boundary_condition=Boundaries.neumann, boundary_function="g")

a_generated_string, L_generated_string = weak_form_object.solve()



# Symbole definieren
f, a = sympy.symbols('f a')

# Gleichung erstellen
equation = sympy.Eq(2 * f, 3 + div(a) + div(grad(2 + a)) * 2 + grad( 3 + div(f) * 7))

# Ausdrücke finden, die entweder "div" oder "grad" enthalten
expressions = equation.find(div)

# Ausdrücke ausgeben
for expr in expressions:
    print(expr)

xx = 2 * div(a)

print(xx.args)
'''



boundaryFunctions = {"curl": "g_curl", "grad": "g_grad", "div": "g_div", "laplacian": "g_lap"}


def stokes_eq():
    stokes = "-phi * Laplacian(u_vec) + div(u_vec) * u_vec + grad(p) = f"

    weak_form_object = Weak_form(trial_function_names=["p"], vector_trial_fuction_names=["u_vec"], test_function_names=["v"], vector_test_function_names=["m_vec"], variable_vectors=["var_vec", "m_vec"], string_equation=stokes, boundary_condition=Boundaries.dirichlet, boundary_function=boundaryFunctions)

    a_generated_string, L_generated_string = weak_form_object.solve()

def stokes_eq2():
    from petsc4py.PETSc import ScalarType
    from dolfinx import fem
    from mpi4py import MPI
    from dolfinx import mesh
    mymesh = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    phi = 2
    f = fem.Constant(mymesh, ScalarType((0,0)))

    #f = fem.Constant(mymesh, [1,1,1])
    stokes = "-phi * Laplacian(u_vec) + div(u_vec) * u_vec + grad(p) = f"

    weak_form_object = Weak_form(functions={"u_vec": {"order": 1, "dim": "vector"}, "p": {"order": 1, "dim": "scalar"}}, mesh="mymesh", string_equation=stokes, boundary_condition=Boundaries.dirichlet, boundary_function=boundaryFunctions)

    a_generated_string, L_generated_string, c = weak_form_object.solve()
    exec(c)

def lin_elas(): 
    lin_el = "-div(sigma) = f"
    weak_form_object = Weak_form(tensor_trial_function_names=["sigma"], vector_test_function_names=["v_vec"], variable_vectors=["f"], string_equation=lin_el, boundary_condition=Boundaries.dirichlet, boundary_function=boundaryFunctions)
    a_generated_string, L_generated_string = weak_form_object.solve()

def rand():
    rand = "grad(p) = f"
    myMesh = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    weak_form_object = Weak_form(functions={"p": {"order": 1, "dim": "skalar"}, "m": {"order": 1, "dim": "vector"}}, mesh="myMesh", trial_function_names=["p"], vector_trial_fuction_names=["u_vec"], test_function_names=["v"], vector_test_function_names=["m_vec"], variables=["m"], variable_vectors=["var_vec", "m_vec"], string_equation=rand, boundary_condition=Boundaries.dirichlet, boundary_function=boundaryFunctions)

    a_generated_string, L_generated_string, commands = weak_form_object.solve()


    exec(commands, globals(), locals())



def skalarproduct():
    skalProd = "inner(grad(p), var_vec) = f"

    weak_form_object = Weak_form(trial_function_names=["p"], vector_trial_fuction_names=["u_vec"], test_function_names=["v"], vector_test_function_names=["m_vec"], variables=["m"], variable_vectors=["var_vec", "m_vec"], string_equation=skalProd, boundary_condition=Boundaries.dirichlet, boundary_function=boundaryFunctions)

    a_generated_string, L_generated_string = weak_form_object.solve()


u_vec_dict = {
    "u": {
        "order": 1,
        "dim": "vector"
    }
}
p_dict = {
    "p": {
        "order": 1,
        "dim": "scalar"
    }
}

u_vec_dict.update(p_dict)

#skalarproduct()
stokes_eq2()
#lin_elas()
#rand()