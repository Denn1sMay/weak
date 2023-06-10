
import sympy
from typing import Optional
from .dimensions.dimensions import Dimensions
from .operators.operators import div, grad, curl, inner
from sympy.vector import Laplacian



def debug_print(debug: bool, message: str, term: Optional[sympy.Expr] = "unknown", format: Optional[str] = "default"):
    if debug == True:
        if format == "heading":
            print(".........." + message + "..........")
            sympy.pprint(term)
        if format == "sympyPprint":
            print(message)
            sympy.pprint(term)
            print(" ")
        if format == "default":
            print("[" + str(term) + "]: " + message)
            

def verify_vector_args(function: sympy.Expr, trial: Optional[sympy.Symbol] = None, trial_vector: Optional[sympy.Symbol] = None, test: Optional[sympy.Symbol] = None, test_vector: Optional[sympy.Symbol] = None):
    first_function_args = function.args[0]
    if (trial != None and first_function_args.has(trial)) or (test != None and first_function_args.has(test)):
        print("Dimension mismatch - expected vector but got skalar")
        print("Incorrect term:")
        sympy.pprint(function)
        raise Exception("Dimension mismatch - expected vector valued trialfunction, but got skalar")
    if trial_vector == None:
        raise Exception("No vector valued trial function provided")


def verify_skalar_args(function: sympy.Expr, trial: Optional[sympy.Symbol] = None, trial_vector: Optional[sympy.Symbol] = None, test: Optional[sympy.Symbol] = None, test_vector: Optional[sympy.Symbol] = None):
    first_function_args = function.args[0]
    if (trial_vector != None and first_function_args.has(trial_vector)) or (test_vector != None and first_function_args.has(test_vector)):
        print("Dimension mismatch - expected skalar but got vector")
        print("Incorrect term:")
        sympy.pprint(function)
        raise Exception("Dimension mismatch - expected skalar valued trialfunction, but got vector")
    if trial == None:
        raise Exception("No skalar valued trial function provided")

def verify_differential_term_has_trial_or_test(function: sympy.Expr, trial: Optional[sympy.Symbol] = None, trial_vector: Optional[sympy.Symbol] = None, test: Optional[sympy.Symbol] = None, test_vector: Optional[sympy.Symbol] = None):
    first_function_args = function.args[0]
    if not first_function_args.has(trial) and not first_function_args.has(trial_vector) and not first_function_args.has(test) and not first_function_args.has(test_vector):
        print("The Expression '" + str(first_function_args) + "' does not contain a trial or test function. Applying differential Operators to this argument will result in 0")

def calculate_dimension(sympy_term: sympy.Expr, trial: Optional[sympy.Symbol] = None, trial_vector: Optional[sympy.Symbol] = None, test: Optional[sympy.Symbol] = None, test_vector: Optional[sympy.Symbol] = None, debug: Optional[bool] = True):
    summand_dimension = None

    if sympy_term.has(div) and not sympy_term.has(grad) and not sympy_term.has(inner) and not sympy_term.has(Laplacian):
        debug_print(debug, "Summand contains a divergence -> skalar valued", sympy_term, "sympyPprint")
        summand_dimension = Dimensions.skalar
        div_atoms = sympy_term.atoms(div)
        verify_vector_args(min(div_atoms), trial, trial_vector, test, test_vector)
        if len(div_atoms) > 1:
            raise Exception("Nested divergences found in one summand - not supported")
            
    if sympy_term.has(grad) and not sympy_term.has(div) and not sympy_term.has(inner) and not sympy_term.has(Laplacian):
        # TODO grad(u) * grad(u) or grad(grad(u)) .... would produce a matrix
        debug_print(debug, "Summand contains a gradient -> vector valued:", sympy_term, "sympyPprint")
        summand_dimension = Dimensions.vector
        grad_atoms = sympy_term.atoms(grad)
        verify_skalar_args(min(grad_atoms), trial, trial_vector, test, test_vector)
        if len(grad_atoms) > 1 and not sympy_term.has(inner):
            raise Exception("Nested gradients found in one summand - probabaly results in a matrix - which is not yet supported")

    if sympy_term.has(grad) and sympy_term.has(inner) and not sympy_term.has(div) and not sympy_term.has(Laplacian):
        debug_print(debug, "Summand contains a inner product -> skalar valued:", sympy_term, "sympyPprint")
        summand_dimension = Dimensions.skalar
        inner_atoms = sympy_term.atoms(inner)
        inner_atom = min(inner_atoms)
        # inner product can look like inner(grad(u), grad(v)) -> Skalar or inner(grad(u_vec), grad(v_vec)) -> Skalar
        # => No Validation of input dimensions required, just have to be present in some form
        first_inner_arg = inner_atom.args[0]
        second_inner_arg = inner_atom.args[1]
        if first_inner_arg.has(grad):
            first_grad_atoms = first_inner_arg.atoms(grad)
            first_grad_atom = min(first_grad_atoms)
            verify_differential_term_has_trial_or_test(first_grad_atom, trial, trial_vector, test, test_vector)
        if second_inner_arg.has(grad):
            second_grad_atoms = second_inner_arg.atoms(grad)
            second_grad_atom = min(second_grad_atoms)
            verify_differential_term_has_trial_or_test(second_grad_atom, trial, trial_vector, test, test_vector)


    if sympy_term.has(grad) and sympy_term.has(div) and not sympy_term.has(Laplacian):
        #only grad(div(..)) is possible -> div(grad(..)) has already been converted to Laplacian(..)
        grad_atoms = sympy_term.atoms(grad)
        grad_atom = min(grad_atoms)
        grad_arg = grad_atom.args[0]
        if grad_arg.has(div):
            # grad(div())
            debug_print(debug, "Summand contains a grad(div(..)) -> vector valued:", sympy_term, "sympyPprint")
            div_atoms = grad_arg.atoms(div)
            div_atom = min(div_atoms)
            verify_vector_args(div_atom, trial, trial_vector, test, test_vector)
            summand_dimension = Dimensions.vector
            

    if sympy_term.has(curl) and not sympy_term.has(grad) and not sympy_term.has(inner) and not sympy_term.has(Laplacian):
        debug_print(debug, "Summand contains a curl -> vektor valued:", sympy_term, "sympyPprint")
        summand_dimension = Dimensions.vector
        curl_atoms = sympy_term.atoms(curl)
        verify_vector_args(min(curl_atoms), trial, trial_vector, test, test_vector)
        if len(curl_atoms) > 1:
            raise Exception("Multiple curls found in one summand - not supported")
        
    if sympy_term.has(Laplacian):
        laplacian_atoms = sympy_term.atoms(Laplacian)
        laplacian_atom = min(laplacian_atoms)
        laplacian_arg = laplacian_atom.args[0]
        if trial != None and laplacian_arg.has(trial):
            debug_print(debug, "Summand contains Laplacian(u_skalar) -> skalar valued:", sympy_term, "sympyPprint")
            summand_dimension = Dimensions.skalar
        elif trial_vector != None and laplacian_arg.has(trial_vector):
            debug_print(debug, "Summand contains Laplacian(u_vector) -> vector valued:", sympy_term, "sympyPprint")
            summand_dimension = Dimensions.vector
        else:
            raise Exception("Laplacian must contain a trial function")

    if summand_dimension == None:
        debug_print(debug, "No differential operator detected - will assume dimension:", sympy_term, "sympyPprint")

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


        