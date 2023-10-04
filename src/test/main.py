import sympy
from ..weakgen.scripts.integral.util.operators.operators import div, grad, curl
from ..weakgen.scripts.integral.util.boundaries.boundaries import Boundaries, BoundaryFunctions
from ..weakgen.scripts.integral.integral import Integral
from ..weakgen.weak_form import Weak_form
from typing import Optional
from mpi4py import MPI
from dolfinx import mesh


from petsc4py.PETSc import ScalarType
from dolfinx import fem
from mpi4py import MPI
from dolfinx import mesh

boundaryFunctions = {"curl": "g_curl", "grad": "g_grad", "div": "g_div", "laplacian": "g_lap"}
mymesh = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)

g_curl = fem.Constant(mymesh, ScalarType((0,0)))
g_grad = fem.Constant(mymesh, ScalarType((0,0)))
g_div = 0
g_lap = 0
def stokes_eq2():
    phi = 2
    f = fem.Constant(mymesh, ScalarType((0,0)))

    #f = fem.Constant(mymesh, [1,1,1])
    vars = {
        "u_vec": {
            "order": 1,
            "dim": "vector"
        }, 
            "p": {
                "order": 1, 
                "dim": "scalar"            
            }
        }
    stokes = "-phi * Laplacian(u_vec) + div(u_vec) * u_vec + grad(p) = f"

    weak_form_object = Weak_form(functions=vars, mesh="mymesh", string_equation=stokes, boundary_condition=Boundaries.dirichlet, boundary_function=boundaryFunctions, debug=False)

    a_generated_string, L_generated_string, c = weak_form_object.solve()
    #exec(c, globals(), locals())

def lin_elas(): 
    vars = {
        "sigma": {
            "dim": "matrix",
            "order": 1,
        },
        "v": {
            "dim": "vector",
            "order": 2
        }
    }

    lin_el = "-div(sigma) = f"
    weak_form_object = Weak_form(functions=vars, mesh=mymesh, string_equation=lin_el, boundary_condition=Boundaries.neumann, boundary_function=boundaryFunctions)
    a_generated_string, L_generated_string, commands = weak_form_object.solve()

def rand():
    vars = {
        "p": 
        {
            "order": 1, 
             "dim": "scalar"
        }, 
        "m": 
        {
            "order": 1, 
            "dim": "vector"
        }
        }
    rand = "grad(p) = f"
    myMesh = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    weak_form_object = Weak_form(functions=vars, mesh="myMesh", variable_vectors=["var_vec", "m_vec"], string_equation=rand, boundary_condition=Boundaries.dirichlet, boundary_function=boundaryFunctions)

    a_generated_string, L_generated_string, commands = weak_form_object.solve()


    #exec(commands, globals(), locals())



def skalarproduct():
    vars= {
        "p": {
            "dim": "scalar",
            "order": 1
        }
    }

    skalProd = "inner(grad(p), var_vec) = f"

    weak_form_object = Weak_form(functions=vars, mesh=mymesh, variables=["m"], variable_vectors=["var_vec", "m_vec"], string_equation=skalProd, boundary_condition=Boundaries.neumann, boundary_function=boundaryFunctions, debug=True)

    a_generated_string, L_generated_string, commands = weak_form_object.solve()

def curlEx():
    vars= {
        "u": {
            "dim": "vector",
            "order": 1
        }
    }

    skalProd = "3*curl(2*u) = f "

    weak_form_object = Weak_form(functions=vars, mesh=mymesh, variables=["m"], variable_vectors=["var_vec", "m_vec"], string_equation=skalProd, boundary_condition=Boundaries.neumann, boundary_function=boundaryFunctions)

    a_generated_string, L_generated_string, commands = weak_form_object.solve()


def gradient_ex():
    vars= {
        "u": {
            "dim": "scalar",
            "order": 1
        },
        "v": {
            "dim": "vector",
            "order": 1
        }
    }

    skalProd = "2*grad(u) = f "

    weak_form_object = Weak_form(functions=vars, 
                                mesh=mymesh, 
                                string_equation=skalProd,
                                boundary_condition=Boundaries.neumann, 
                                boundary_function=boundaryFunctions)

    a_generated_string, L_generated_string, commands = weak_form_object.solve()

def divergence_ex():
    vars= {
        "u": {
            "dim": "matrix",
            "order": 1
        },
        "v": {
            "dim": "vector",
            "order": 1
        }
    }

    skalProd = "div(2*u) = f "

    weak_form_object = Weak_form(functions=vars, 
                                mesh=mymesh, 
                                string_equation=skalProd,
                                boundary_condition=Boundaries.neumann, 
                                boundary_function=boundaryFunctions)

    a_generated_string, L_generated_string, commands = weak_form_object.solve()

def laplace_ex():
    vars= {
        "u": {
            "dim": "vector",
            "order": 1
        },
        "v": {
            "dim": "vector",
            "order": 1
        }
    }

    skalProd = "Laplacian(u * 2) = f "

    weak_form_object = Weak_form(functions=vars, 
                                mesh=mymesh, 
                                string_equation=skalProd,
                                boundary_condition=Boundaries.neumann, 
                                boundary_function=boundaryFunctions)

    a_generated_string, L_generated_string, commands = weak_form_object.solve()

def someRandomEquation():
    vars= {
        "u_vec": {
            "dim": "vector",
            "order": 1
        },
        "p": {
            "dim": "scalar",
            "order": 1
        },
        "m_vec": {
            "dim": "vector",
            "order": 2
        }
    }

    pi = 3.14
    constantVec = fem.Constant(mymesh, ScalarType((2,2)))

    random_eq = "5 * grad(p) + pi + div(m_vec + constantVec) + curl(2*u_vec) = constantVec"

    weak_form_object = Weak_form(functions=vars, 
                                mesh="mymesh", 
                                string_equation=random_eq, 
                                boundary_condition=Boundaries.dirichlet, 
                                boundary_function=boundaryFunctions,
                                variables=["pi"],
                                variable_vectors=["constantVec"],
                                debug=False)
    a_generated_string, L_generated_string, commands = weak_form_object.solve()



someRandomEquation()
laplace_ex()
gradient_ex()
divergence_ex()
skalarproduct()
stokes_eq2()
lin_elas()
rand()
curlEx()