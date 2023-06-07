
import sympy
from typing import Optional
from .dimensions.dimensions import Dimensions
from .operators.operators import div, grad, curl, inner
from sympy.vector import Laplacian

def verify_vector_args(function: sympy.Expr, trial: Optional[sympy.Symbol] = None, trial_vector: Optional[sympy.Symbol] = None):
    first_function_args = function.args[0]
    if trial != None and first_function_args.has(trial):
        print("Dimension mismatch - expected vector but got skalar")
        print("Incorrect term:")
        sympy.pprint(function)
        raise Exception("Dimension mismatch - expected vector valued trialfunction, but got skalar")


def verify_skalar_args(function: sympy.Expr, trial: Optional[sympy.Symbol] = None, trial_vector: Optional[sympy.Symbol] = None):
    first_function_args = function.args[0]
    print("verify skalar value of ..........")
    sympy.pprint(function)
    if trial_vector != None and first_function_args.has(trial_vector):
        print("Dimension mismatch - expected skalar but got vector")
        print("Incorrect term:")
        sympy.pprint(function)
        raise Exception("Dimension mismatch - expected skalar valued trialfunction, but got vector")


def calculate_dimension(sympy_term: sympy.Expr, trial: Optional[sympy.Symbol] = None, trial_vector: Optional[sympy.Symbol] = None):
    summand_dimension = None
    if sympy_term.has(div) and not sympy_term.has(grad):
        # The expression is probably skalar valued
        # but if there are other differential operators in the term it could also be vector valued
        print("Summand contains a divergence -> skalar valued")
        print(sympy_term)
        summand_dimension = Dimensions.skalar
        div_atoms = sympy_term.atoms(div)
        verify_vector_args(min(div_atoms), trial, trial_vector)
        if len(div_atoms) > 1:
            raise Exception("Multiple divergences found in one summand - not supported")
            
    if sympy_term.has(grad) and not sympy_term.has(div) and not sympy_term.has(inner):
        # The expression is probalby vector valued
        # but if contained with an inner product it becomes skalar valued
        # but if there is another gradient inside the gradient (or a vecotr), then is could be a matrix
        # TODO grad(u) * grad(u) or grad(grad(u)) .... would produce a matrix
        print("Summand contains a gradient -> vector valued")
        sympy.pprint(sympy_term)
        summand_dimension = Dimensions.vector
        if sympy_term.has(inner):
            print("Summand contains a gradient and inner function -> skalar valued")
            summand_dimension = Dimensions.skalar
        grad_atoms = sympy_term.atoms(grad)
        verify_skalar_args(min(grad_atoms), trial, trial_vector)
        if len(grad_atoms) > 1 and not sympy_term.has(inner):
            raise Exception("Multiple gradients found in one summand - probabaly results in a matrix - which is not yet supported")

    if sympy_term.has(curl):
        # the expression is probably vector valued
        # but if there are othr differential operators 
        print("Summand contains a curl -> vektor valued")
        print(sympy_term)
        summand_dimension = Dimensions.vector
        curl_atoms = sympy_term.atoms(curl)
        verify_vector_args(min(curl_atoms), trial, trial_vector)
        if len(curl_atoms) > 1:
            raise Exception("Multiple curls found in one summand - not supported")
        
    if sympy_term.has(div) and sympy_term.has(grad):
        print("Summand contains div and grad -> Laplace -> skalar valued")
        summand_dimension = Dimensions.skalar

    if sympy_term.has(Laplacian):
        print("Summand contains Laplace -> skalar valued")
        summand_dimension = Dimensions.skalar
        laplace_atoms = sympy_term.atoms(Laplacian)
        verify_skalar_args(min(laplace_atoms), trial, trial_vector)

    if summand_dimension == None:
        print("No differential operator detected - should use default dimension")
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
    
def replace_div_grad_with_laplace(term: sympy.Expr):
    returned_term = term
    if term.has(div):
        div_functions = term.atoms(div)
        div_function = min(div_functions)
        div_args = div_function.args
        first_div_arg = div_args[0]
        if first_div_arg.has(grad):
            grad_functions = first_div_arg.atoms(grad)
            grad_function = min(grad_functions)
            grad_args = grad_function.args
            first_grad_arg = grad_args[0]
            returned_term = term.subs(div_function, Laplacian(first_grad_arg))
    return returned_term


        