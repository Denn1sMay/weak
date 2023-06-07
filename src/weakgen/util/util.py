import sympy
from typing import Optional, List
from ..scripts.integral.integral import Integral
from ..scripts.integral.util.boundaries.boundaries import Boundaries
from typing import Literal

_side_types = ["lhs", "rhs"]
def sort_terms(terms: List[sympy.Expr], side: _side_types = "lhs", trial: Optional[sympy.Symbol] = None, test: Optional[sympy.Symbol] = None, trial_vector: Optional[sympy.Symbol] = None, test_vector: Optional[sympy.Symbol] = None, boundary: Optional[Boundaries] = None, boundary_func: Optional[sympy.Symbol] = None):
        new_lhs_terms = []
        new_rhs_terms = []
        rhs_factor = 1
        lhs_factor = 1
        if side == "lhs":
            rhs_factor = -1
        if side == "rhs":
            lhs_factor = -1
        # TODO if lhs term is added to rhs, the Vorzeichen has to be inverted
        for term in terms:
            if trial != None and trial_vector != None:
                if term.has(trial) or term.has(trial_vector):
                    new_lhs_terms.append(Integral(lhs_factor * term, trial=trial, test=test, trial_vector=trial_vector, test_vector=test_vector, boundary_condition=boundary, boundary_function=boundary_func))
                else:
                    new_rhs_terms.append(Integral(rhs_factor * term, trial=trial, test=test, trial_vector=trial_vector, test_vector=test_vector, boundary_condition=boundary, boundary_function=boundary_func))
            elif trial != None and trial_vector == None:
                if term.has(trial):
                    new_lhs_terms.append(Integral(lhs_factor * term, trial=trial, test=test, boundary_condition=boundary, boundary_function=boundary_func))
                else:
                    new_rhs_terms.append(Integral(rhs_factor * term, trial=trial, test=test, boundary_condition=boundary, boundary_function=boundary_func))
            elif trial_vector != None and trial == None:
                if term.has(trial_vector):
                    new_lhs_terms.append(Integral(lhs_factor * term, trial_vector=trial_vector, test_vector=test_vector, boundary_condition=boundary, boundary_function=boundary_func))
                else:
                    new_rhs_terms.append(Integral(rhs_factor * term, trial_vector=trial_vector, test_vector=test_vector, boundary_condition=boundary, boundary_function=boundary_func))
            else:
                raise Exception("Need to provide string literals of trial- and test function(s)")
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
    lhs = ""
    for index, term in enumerate(terms):
        lhs = lhs + ("" if index == 0 else " + ") + term.convert_integral_to_ufl_string()
    return lhs
