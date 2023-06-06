
import sympy
from .dimensions.dimensions import Dimensions
from .operators.operators import div, grad, rot, inner

def calculate_dimension(sympy_term: sympy.Expr, trial_function_u: sympy.Symbol, test_function_v: sympy.Symbol):
    summand_dimension = Dimensions.skalar
    if sympy_term.has(div):
        # The expression is probably skalar valued
        # but if there are other differential operators in the term it could also be vector valued
        print("Summand contains a divergence -> skalar valued")
        print(sympy_term)
        summand_dimension = Dimensions.skalar
        div_atoms = sympy_term.atoms(div)
        if len(div_atoms) > 1:
            raise Exception("Multiple divergences found in one summand - not supported")
        if sympy_term.has(grad):
            if sympy_term.has(div(grad(trial_function_u))):
                print("Divergence of gradient found in term -> equals laplace operator")
            else:
                raise Exception("div and grad found inside of term, but cannot be converted to laplace expression")
            
    if sympy_term.has(grad):
        # The expression is probalby vector valued
        # but if contained with an inner product it becomes skalar valued
        # but if there is another gradient inside the gradient (or a vecotr), then is could be a matrix
        # TODO grad(u) * grad(u) or grad(grad(u)) .... would produce a matrix
        print("Summand contains a gradient -> vector valued")
        sympy.pprint(sympy_term)
        summand_dimension = Dimensions.vector
        grad_atoms = sympy_term.atoms(grad)
        if len(grad_atoms) > 1 and not sympy_term.has(inner):
            raise Exception("Multiple gradients found in one summand - probabaly results in a matrix - which is not yet supported")

    if sympy_term.has(rot):
        # the expression is probably vector valued
        # but if there are othr differential operators 
        print("Summand contains a rotation -> vektor valued")
        print(sympy_term)
        summand_dimension = Dimensions.vector
        rot_atoms = sympy_term.atoms(rot)
        if len(rot_atoms) > 1:
            raise Exception("Multiple rotations found in one summand - not supported")
    return summand_dimension



def multiply(terms: list):
    multiplied_terms = []
    for term in terms:
        multiplied_terms.append(term.multiply_with_test_function().term)
    return multiplied_terms



#TODO expression can contain multiple divergence operators
def get_differential_function(searched_function: sympy.Function, term: sympy.Expr):
    function_expr = term.atoms(searched_function)
    if len(function_expr) > 1:
        raise Exception("Nested differential operator detected - not supported")
    first_function_atom = min(function_expr)
    function_args = first_function_atom.args
    function_args_with_trial = function_args[0]
    return first_function_atom, function_args_with_trial
    
