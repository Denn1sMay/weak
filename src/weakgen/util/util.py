import sympy
from typing import Optional, List
from ..scripts.integral.integral import Integral
from ..scripts.integral.util.boundaries.boundaries import Boundaries
from ..scripts.integral.util.dimensions.dimensions import Dimensions
from typing import Literal
from dolfinx import mesh

def verify_dict(variables: dict):
    for key, value in variables.items():
        if "dim" in value:
            if value["dim"] != "scalar" and value["dim"] != "vector" and value["dim"] != "matrix" and value["dim"] != "tensor" and value["dim"] != Dimensions.scalar and value["dim"] != Dimensions.vector and value["dim"] != Dimensions.matrix:
                raise Exception(f"Invalid 'dim' type in function '{key}' - please provide 'scalar', 'vector' or 'tensor'.")
        else:
            raise Exception(f"No 'dim' key found in function '{key}' - please provide a dimension.")

def get_sympy_symbols(variables: dict):
    scalar_dicts = {}
    vector_dicts = {}
    tensor_dicts = {}
    scalar_function_symbols = []
    vector_function_symbols = []
    tensor_function_symbols = []
    scalar_test_function_symbols = []
    vector_test_function_symbols = []
    for key, value in variables.items():
        print(value)
        if (value.get("dim") == "scalar" or value.get("dim") == Dimensions.scalar):
            scalar_dicts.update({key: value})
            scalar_function_symbols.append(sympy.Symbol(key))
            scalar_test_function_symbols.append(sympy.Symbol(key + "_test"))

        if (value.get("dim") == "vector" or value.get("dim") == Dimensions.vector):
            vector_dicts.update({key: value})
            vector_function_symbols.append(sympy.Symbol(key))
            vector_test_function_symbols.append(sympy.Symbol(key + "_test"))

        if (value.get("dim") == "tensor" or value.get("dim") == "matrix" or value.get("dim") == Dimensions.matrix):
            tensor_dicts.update({key: value})
            tensor_function_symbols.append(sympy.Symbol(key))

    return scalar_function_symbols, vector_function_symbols, tensor_function_symbols, scalar_test_function_symbols, vector_test_function_symbols, scalar_dicts, vector_dicts, tensor_dicts

_side_types = ["lhs", "rhs"]
def sort_terms(terms: List[sympy.Expr], side: _side_types, trial: Optional[List[sympy.Symbol]] = None, test: Optional[List[sympy.Symbol]] = None, trial_vector: Optional[List[sympy.Symbol]] = None, test_vector: Optional[List[sympy.Symbol]] = None, trial_tensor: Optional[List[sympy.Symbol]] = None, variables: Optional[List[sympy.Symbol]] = None, variable_vectors: Optional[List[sympy.Symbol]] = None,  boundary: Optional[Boundaries] = None, boundary_func: Optional[sympy.Symbol] = None, debug: Optional[bool] = True):
        new_lhs_terms = []
        new_rhs_terms = []
        rhs_factor = 1
        lhs_factor = 1
        if side == "lhs":
            rhs_factor = -1
        if side == "rhs":
            lhs_factor = -1
        for term in terms:
            if (trial != None and term.has(*trial)) or (trial_vector != None and term.has(*trial_vector) or (trial_tensor != None and term.has(*trial_tensor))):
                new_lhs_terms.append(Integral(lhs_factor * term, trial=trial, test=test, trial_vector=trial_vector, test_vector=test_vector, trial_tensor=trial_tensor, variables=variables, variable_vectors=variable_vectors, boundary_condition=boundary, boundary_function=boundary_func, debug=debug))
            else:
                new_rhs_terms.append(Integral(rhs_factor * term, trial=trial, test=test, trial_vector=trial_vector, test_vector=test_vector, trial_tensor=trial_tensor, variables=variables, variable_vectors=variable_vectors, boundary_condition=boundary, boundary_function=boundary_func, debug=debug))
        return new_lhs_terms, new_rhs_terms

def execute_test_multiplications(terms: List[Integral]):
    multiplied_terms = []
    for term in terms:
        term.multiply_with_test_function()
        multiplied_terms.append(term)
    return multiplied_terms

def execute_integration(terms: List[Integral]):
    integrated_terms = []
    for term in terms:
        term.integrate_over_domain()
        integrated_terms.append(term)
    return integrated_terms

def execute_integration_by_parts(terms: List[Integral]):
    partially_integrated_terms = []
    for term in terms:
        term.integrate_by_parts()
        partially_integrated_terms.append(term)
    return partially_integrated_terms

def execute_ufl_conversion(terms: List[Integral]):
    converted_to_ufl = ""
    for index, term in enumerate(terms):
        converted_to_ufl = converted_to_ufl + ("" if index == 0 else " + ") + term.convert_integral_to_ufl_string()
    return converted_to_ufl


def get_executable_string(variables: dict, mesh, lhs, rhs):
    imports = """
from ufl import FiniteElement, VectorElement, MixedElement, TestFunctions, TrialFunctions, TrialFunction, TestFunction, inner, dot, grad, div, curl, div, ds, dx
from dolfinx.fem import FunctionSpace
from mpi4py import MPI
from dolfinx import mesh
    """

    space_name = "V"
    if len(variables.items()) == 1:
        for key, values in variables.items():
            space_name = values["spaceName"] if "spaceName" in values else space_name

    elements = ""
    counter = 1
    elements_string = ""
    for key, value in variables.items():
        if (value["dim"] == "scalar" or value["dim"] == Dimensions.scalar):
            elements = elements + f"""
fe_{str(counter)} = FiniteElement('Lagrange', {mesh}.ufl_cell(), {value["order"] if "order" in value else 1})
""" 
        elif (value["dim"] == "vector" or value["dim"] == Dimensions.vector):
            elements = elements + f"""
fe_{str(counter)} = VectorElement('Lagrange', {mesh}.ufl_cell(), {value["order"] if "order" in value else 1})
""" 
        elements_string = elements_string + f"fe_{str(counter)}"
        elements_string = elements_string + ", " if counter < len(variables.items()) else elements_string
        counter = counter + 1


    mixed_element = f"""
mixed_elem = MixedElement({elements_string})
""" if counter > 2 else f"""
mixed_elem = fe_1
"""

    function_space = f"""
{space_name} = FunctionSpace({mesh}, mixed_elem)
"""
    function_spaces = ""
    counter = 1
    for key, value in variables.items():
        if len(variables.items()) > 1:
            sub_space_name = value["spaceName"] if "spaceName" in value else f"V_{counter}"
            function_spaces = function_spaces + f"""
{sub_space_name} = {space_name}.sub({counter-1})
"""

    trial_functions = ""
    test_functions = ""
    counter = 1
    for key, value in variables.items():
        trial_functions = trial_functions + key
        trial_functions = trial_functions + ", " if counter < len(variables.items()) else trial_functions

        test_functions = test_functions + key + "_test"
        test_functions = test_functions + ", " if counter < len(variables.items()) else test_functions
        counter = counter + 1

    function_decl = f"""
{trial_functions} = TrialFunctions({space_name})
""" if counter > 2 else f"""
{trial_functions} = TrialFunction({space_name})
"""
    test_function_decl = f"""
{test_functions} = TestFunctions({space_name})
""" if counter > 2 else f"""
{test_functions} = TestFunction({space_name})
"""
    L_decl = f"""
L = {rhs}
"""
    L_decl = "L = (dot(f, u_vec_test)) * dx"
    a_decl = f"""
a = {lhs}
"""

    concated = imports + elements + mixed_element + function_space + function_spaces + function_decl + test_function_decl + a_decl + L_decl
    print(concated)
    return concated

