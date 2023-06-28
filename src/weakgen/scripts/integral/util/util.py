
import sympy
from typing import Optional, List
from .dimensions.dimensions import Dimensions
from .operators.operators import div, grad, curl, inner
from sympy.vector import Laplacian



def get_expression_types(expression: sympy.Expr, type: Optional[str] = "add", productId: Optional[int] = 0):
    if expression.func == sympy.Mul:
        typed_expressions = []
        for arg in expression.args:
            diff_func = get_expression_types(arg, "mul", productId)
            typed_expressions = typed_expressions + diff_func
        return typed_expressions

    if expression.func == sympy.Add:
        typed_expressions = []
        productId = productId + 1
        for arg in expression.args:
            diff_func = get_expression_types(arg, "add", productId)
            typed_expressions = typed_expressions + diff_func
        return typed_expressions
    
    if expression.func == div:
        return [{"expression": expression, "operator": div, "type": type, "productId": productId}]
    if expression.func == grad:
        return [{"expression": expression, "operator": grad, "type": type, "productId": productId}]
    if expression.func == curl:
        return [{"expression": expression,"operator": curl, "type": type, "productId": productId}]
    if expression.func == inner:
        return [{"expression": expression,"operator": inner, "type": type, "productId": productId}]
    if expression.func == Laplacian:
        return [{"expression": expression, "operator": Laplacian, "type": type, "productId": productId}]
    if expression.func == sympy.Symbol:
        return [{"expression": expression, "operator": "symbol", "type": type, "productId": productId}]
    if expression.func == sympy.Integer:
        return [{"expression": expression, "operator": "integer", "type": type, "productId": productId}]
    return [{"expression": expression, "operator": "unknown", "type": type, "productId": productId}]


'''
Will apply rules on how the dimensions are shifted by using nabla operator.
The 'expand()' operator is used on the expression -> only summands should be present in typed_expressions
'''
def get_dimension(expression: sympy.Expr, trial: Optional[List[sympy.Symbol]] = [], trial_vector: Optional[List[sympy.Symbol]] = [], test: Optional[List[sympy.Symbol]] = [], test_vector: Optional[List[sympy.Symbol]] = [], variables: Optional[List[sympy.Symbol]] = [], variable_vectors: Optional[List[sympy.Symbol]] = [], currentDimension: Optional[int] = None):
    print("NEW INIT ___________________________________")
    typed_expressions = get_expression_types(expression.expand())
    grouped_products = typed_expressions
    dimension = 0
    expected_dimension = []
    default_dimension = None
    for typed_e in typed_expressions:
        print(typed_e)
        if typed_e["operator"] == div:
            div_arg = typed_e["expression"].args[0]
            dimension_shift = get_dimension(div_arg, trial, trial_vector, test, test_vector, variables, variable_vectors, dimension)
            print("DIMENSION SHIFT")
            print(dimension_shift)
            expected_dimension.append(dimension_shift - 1)

        if typed_e["operator"] == grad:
            grad_arg = typed_e["expression"].args[0]
            dimension_shift = get_dimension(grad_arg, trial, trial_vector, test, test_vector, variables, variable_vectors, dimension)
            expected_dimension.append(dimension_shift + 1)

        if typed_e["operator"] == inner:
            first_arg = typed_e["expression"].args[0]
            second_arg = typed_e["expression"].args[1]
            dimension_shift_first = get_dimension(first_arg, trial, trial_vector, test, test_vector, variables, variable_vectors, dimension)
            dimension_shift_second = get_dimension(second_arg, trial, trial_vector, test, test_vector, variables, variable_vectors, dimension)
            higher_dim = max([dimension_shift_first, dimension_shift_second])
            if abs(abs(dimension_shift_first) - abs(dimension_shift_second)) > 1:
                print("Invalid Inner Product:")
                sympy.pprint(typed_e["operator"])
                raise Exception("Cannot take inner product of expression with dimension difference > 1")
            dimension = dimension + dimension_shift_first
            expected_dimension.append(higher_dim - 1)

        if typed_e["operator"] == curl:
            curl_arg = typed_e["expression"].args[0]
            dimension_shift = get_dimension(curl_arg, trial, trial_vector, test, test_vector, variables, variable_vectors, dimension)
            expected_dimension.append(dimension_shift)

        if typed_e["operator"] == Laplacian:
            lap_arg = typed_e["expression"].args[0]
            dimension_shift = get_dimension(lap_arg, trial, trial_vector, test, test_vector, variables, variable_vectors, dimension)
            expected_dimension.append(dimension_shift)

        if typed_e["operator"] == "symbol" and typed_e["type"] == "add":
            if (typed_e["expression"] in trial_vector) or (typed_e["expression"] in test_vector) or (typed_e["expression"] in variable_vectors):
                default_dimension = 1
            elif typed_e["expression"] in trial or typed_e["expression"] in test or typed_e["expression"] in variables:
                default_dimension = 0
            else:
                # Unknown variable - assuming skalar value
                if currentDimension != None and currentDimension == -1:
                    # Current dimension is -1 means this unknown variable should be a vector
                    if default_dimension != None and default_dimension != 1:
                        print(typed_e["expression"])
                        raise Exception("Cannot determine dimension - expected vector value")
                    # Assume Vector value and continue
                    default_dimension = 1
                else:
                    # Assume skalar value
                    if default_dimension == None:
                        default_dimension = 0

        if typed_e["operator"] == "integer" and typed_e["type"] == "add":
            default_dimension = 0
        
    
    if len(expected_dimension) > 0 and (all(p == expected_dimension[0] for p in expected_dimension) == False or default_dimension != None and expected_dimension[0] != default_dimension):
        print("A Dimension mismatch in the following expression has been detected - cannot execute transformation")
        sympy.pprint(expression)
        raise Exception("Different dimensions detected in expression")
    else:
        dimension = expected_dimension[0] if len(expected_dimension) > 0 else 0


    # currentDimension will only contain values when triggered recursively
    if currentDimension != None:
        #TODO CHECK: e.g. div(u) + v would return v dimension
        #-> defaultdimension == dimension
        print("Returned Dimension: ....................")
        print(dimension if default_dimension is None else default_dimension)
        print(default_dimension)
        return dimension if default_dimension is None else default_dimension
    
    else:
        return dimension

    


x = sympy.Symbol("x")
m = sympy.Symbol("m")
k = sympy.Symbol("k")
xx =  grad(m) + grad(inner(grad(m), m))
ll = [x, k]
di = get_dimension(xx, trial=ll, trial_vector=[m])
print("Final Dim")
print(di)



exp = div(x) + m + 3 * x
#print(get_typed_expressions(exp, None))

vv = div(x) + (m + 2 * (x - 1) / m)
#print(get_typed_expressions(vv, None))


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
            

def verify_vector_args(function: sympy.Expr, trial: Optional[List[sympy.Symbol]] = None, trial_vector: Optional[List[sympy.Symbol]] = None, test: Optional[List[sympy.Symbol]] = None, test_vector: Optional[List[sympy.Symbol]] = None):
    first_function_args = function.args[0]
    if (trial != None and first_function_args.has(*trial)) or (test != None and first_function_args.has(*test)):
        print("Dimension mismatch - expected vector but got skalar")
        print("Incorrect term:")
        sympy.pprint(function)
        raise Exception("Dimension mismatch - expected vector valued trialfunction, but got skalar")
    if trial_vector == None and test_vector == None:
        raise Exception("No vector valued function provided")


def verify_skalar_args(function: sympy.Expr, trial: Optional[List[sympy.Symbol]] = None, trial_vector: Optional[List[sympy.Symbol]] = None, test: Optional[List[sympy.Symbol]] = None, test_vector: Optional[List[sympy.Symbol]] = None):
    first_function_args = function.args[0]
    if (trial_vector != None and first_function_args.has(*trial_vector)) or (test_vector != None and first_function_args.has(*test_vector)):
        print("Dimension mismatch - expected skalar but got vector")
        print("Incorrect term:")
        sympy.pprint(function)
        raise Exception("Dimension mismatch - expected skalar valued trialfunction, but got vector")
    if trial == None and test == None:
        raise Exception("No skalar valued trial function provided")

def verify_differential_term_has_trial_or_test(function: sympy.Expr, trial: Optional[List[sympy.Symbol]] = None, trial_vector: Optional[List[sympy.Symbol]] = None, test: Optional[List[sympy.Symbol]] = None, test_vector: Optional[List[sympy.Symbol]] = None):
    first_function_args = function.args[0]
    if (trial != None and not first_function_args.has(*trial)) and (trial_vector != None and not first_function_args.has(*trial_vector)) and (test != None and not first_function_args.has(*test)) and (test_vector != None and not first_function_args.has(*test_vector)):
        print("The Expression '" + str(first_function_args) + "' does not contain a trial or test function. Applying differential Operators to this argument will result in 0")

def calculate_dimension(sympy_term: sympy.Expr, trial: Optional[List[sympy.Symbol]] = None, trial_vector: Optional[List[sympy.Symbol]] = None, test: Optional[List[sympy.Symbol]] = None, test_vector: Optional[List[sympy.Symbol]] = None, debug: Optional[bool] = True):
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
        if trial != None and laplacian_arg.has(*trial):
            debug_print(debug, "Summand contains Laplacian(u_skalar) -> skalar valued:", sympy_term, "sympyPprint")
            summand_dimension = Dimensions.skalar
        elif trial_vector != None and laplacian_arg.has(*trial_vector):
            debug_print(debug, "Summand contains Laplacian(u_vector) -> vector valued:", sympy_term, "sympyPprint")
            summand_dimension = Dimensions.vector
        else:
            raise Exception("Laplacian must contain a trial function")
        
    if not sympy_term.has(Laplacian) and not sympy_term.has(div) and not sympy_term.has(grad)  and not sympy_term.has(curl):
        if (test != None and sympy_term.has(*trial)) or (test != None and sympy_term.has(*test)):
            debug_print(debug, "Summand contains no differential operator -> default skalar dimension:", sympy_term, "sympyPprint")
            summand_dimension = Dimensions.skalar
        elif (trial_vector != None and sympy_term.has(*trial_vector)) or (test_vector != None and sympy_term.has(*test_vector)):
            debug_print(debug, "Summand contains no differential operator -> default vector dimension:", sympy_term, "sympyPprint")
            summand_dimension = Dimensions.vector

    if summand_dimension == None:
        debug_print(debug, "No differential operator detected - will assume dimension:", sympy_term, "sympyPprint")

    return summand_dimension



def multiply(terms: list):
    multiplied_terms = []
    for term in terms:
        multiplied_terms.append(term.multiply_with_test_function().term)
    return multiplied_terms



#TODO expression can contain multiple divergence operators
def get_expression_of_type(searched_function: sympy.Function, term: sympy.Expr):
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


def get_corresponding_test_function(term: sympy.Expr, test: List[sympy.Symbol], trials: Optional[List[sympy.Symbol]] = None):
    if trials == None:
        # No Trial function present - will return the first Test Function in the list
        return test[0]
    
    corresponding_function = None
    for index, trial in enumerate(trials):
        # Will immediately return after first trial function is found (Multiple Trial Functions in term?)
        if term.has(trial):
            corresponding_function = test[index]
    return corresponding_function if corresponding_function != None else test[0]


    
    
