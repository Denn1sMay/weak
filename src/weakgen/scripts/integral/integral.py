import sympy
from sympy.vector import Laplacian
from .util.dimensions.dimensions import Dimensions
from .util.boundaries.boundaries import Boundaries, BoundaryFunctions
from .util.operators.operators import div, grad, curl, inner
from .util.util import get_expression_of_type, get_expression_types, get_sorted_inner_args, replace_div_grad_with_laplace, is_test_inner_product, debug_print, get_corresponding_test_function, get_dimension_type, get_inner, contains_function_on_surface, operator_types
from typing import Optional, List


domain = sympy.Symbol("omega")
surface = sympy.Symbol("surface")

class Integral:
    def __init__(self, term: sympy.Expr, trial: Optional[List[sympy.Symbol]] = [], test: Optional[List[sympy.Symbol]] = [], trial_vector: Optional[List[sympy.Symbol]] = [], test_vector: Optional[List[sympy.Symbol]] = [], trial_tensor: Optional[List[sympy.Symbol]] = [], variables: Optional[List[sympy.Symbol]] = [], variable_vectors: Optional[List[sympy.Symbol]] = [], boundary_condition: Optional[Boundaries] = None, boundary_function: Optional[dict[str,str]] = None, debug: Optional[bool] = False):
        replaced_term = replace_div_grad_with_laplace(term)
        self.term = replaced_term
        self.trial = trial
        self.test = get_corresponding_test_function(term, test, trial) if test != None else test
        self.all_test = test
        self.boundary_condition = boundary_condition
        self.boundary_function = boundary_function
        self.test_vector = get_corresponding_test_function(term, test_vector, trial_vector) if test_vector != None else test_vector
        self.all_test_vectors = test_vector
        self.trial_vector = trial_vector
        self.trial_tensor = trial_tensor
        self.variables = variables
        self.variable_vectors = variable_vectors
        self.boundary_condition = boundary_condition
        self.boundary_function = boundary_function
        self.debug = debug

        self.dim = get_dimension_type(self.term, self.trial, self.trial_vector, self.all_test, self.all_test_vectors, self.trial_tensor, self.variables, self.variable_vectors, debug=self.debug)
        self.is_nonlinear = self.is_nonlinear()


    def loop_inner_expressions(self, expression: sympy.Expr):
        multiplication_executed = False
        current_term = self.term
        replaced_expr = None
        new_expr = None
        while multiplication_executed == False:
            if contains_function_on_surface(inner, current_term):
                inner_func, inner_args = get_inner(self.term)
                for arg in inner_args:
                    for operator in operator_types:
                        if contains_function_on_surface(operator, arg):
                            replaced_expr = arg
                            new_expr = arg * self.test
                            multiplication_executed = True
                            print("RETURNING")
                            return replaced_expr, new_expr
            else:
                break
        return replaced_expr, new_expr
 

    def multiply_with_test_function(self):
        '''
        replaced_expr, new_expr = self.loop_inner_expressions(self.term)
        if replaced_expr != None and new_expr != None:
            self.term = self.term.subs(replaced_expr, new_expr)
            return
        '''
        print("MULTIPLICATION")
        print(self.term)
        print(self.dim)
        if self.dim == Dimensions.skalar:
            print("SKALAR")
            if self.test == None:
                raise Exception("Need to provide string literal for skalar valued test function in order to create inner product")
            self.term = self.term * self.test
        elif self.dim == Dimensions.vector:
            print("VECTOR")
            if self.test_vector == None:
                raise Exception("Need to provide string literal for vector valued test function in order to create inner product")
            self.term = inner(self.term, self.test_vector)
        else:
            self.term = self.term * self.test
        self.dim = get_dimension_type(self.term, self.trial, self.trial_vector, self.all_test, self.all_test_vectors, self.variables, self.variable_vectors, debug=self.debug)



    def integrate_over_domain(self):
        self.term = sympy.Integral(self.term, domain)

    '''
    Integration by Parts
    '''
    def integrate_by_parts(self, currentTerm: Optional[sympy.Expr] = None):
        currentTerm = currentTerm if currentTerm is not None else self.term


        if currentTerm is None and contains_function_on_surface(inner, currentTerm) and (currentTerm.has(div) or currentTerm.has(Laplacian)):
            # Divergence/ Laplacian that returns a vector -> args of operators are matrices
            inner_func, inner_args = get_inner(self.term)
            new_args = []
            for arg in inner_args:
                if arg.has(div) or arg.has(Laplacian):
                    print("INNER")
                    #integrated_parts = self.integrate_by_parts(arg)
                    #new_args.append()

        if(self.is_nonlinear):
            return
        if self.term.has(div) and not self.term.has(grad):
            self.check_linearity(div)
            if self.term.has(Laplacian):
                raise Exception("Divergence and Laplace cannot be combined")
            self.perform_integration_by_parts_on_diveregence()
        elif self.term.has(Laplacian):
            self.check_linearity(Laplacian)
            self.perform_integration_by_parts_on_laplacian()
        elif self.term.has(grad) and not self.term.has(div):
            self.check_linearity(grad)
            self.perform_integration_by_parts_on_gradient()
        elif self.term.has(curl):
            self.check_linearity(curl)
            self.perform_integration_by_parts_on_curl()
        else:
            debug_print(self.debug, "No differential operator present - skipping integration by parts", self.term)
            
    def perform_integration_by_parts_on_curl(self):
        integral = self.term
        debug_print(self.debug, "Performing integration by parts on curl integral", self.term, "heading")
        # Get the curl function
        integral_args = integral.args[0]
        #TODO expression can contain multiple curl operators
        curl_function, args_with_trial = get_expression_of_type(curl, integral_args)
        # This expression will be replaced
        curl_test_inner_product = inner(curl_function, self.test_vector)
        # v must be vector valued
        test_curl = curl(self.test_vector)
        # This expression will be inserted
        trial_curl_inner = inner(args_with_trial, test_curl)
        integral_over_domain = integral_args.subs(curl_test_inner_product, trial_curl_inner)
        integrated_parts = sympy.Integral(integral_over_domain, domain)

        if self.boundary_condition != None:
            boundary_term = self.get_boundary_term(div, args_with_trial)
            if boundary_term != None:
                integrated_parts = integrated_parts - boundary_term

        self.term = integrated_parts
        debug_print(self.debug, "Transformed Integral: ", self.term, "sympyPprint")
        return integrated_parts
      

    def perform_integration_by_parts_on_diveregence(self):
        integral = self.term
        debug_print(self.debug, "Performing integration by parts on divergence integral", self.term, "heading")
        integral_args = integral.args[0]
        divergence_function, args_with_trial = get_expression_of_type(div, integral_args)

        replaced_expression = None
        new_expression = None
        if is_test_inner_product(integral_args, self.test_vector):
            test_gradient = grad(self.test_vector)
            print("_____---__---_--_--_-__-_-_-")
            replaced_expression, inner_args = get_inner(integral_args)
            arg_with_trial, arg_with_test_vector =  get_sorted_inner_args(integral_args, self.test_vector)
            inner_func = inner(args_with_trial, grad(arg_with_test_vector))
            new_expression = arg_with_trial.subs(divergence_function, inner_func)
            print(new_expression)
        else:
            test_gradient = grad(self.test)
            replaced_expression = divergence_function * self.test
            new_expression = inner(args_with_trial, test_gradient)

        integral_over_domain = integral_args.subs(replaced_expression, new_expression)
        integrated_parts = sympy.Integral(integral_over_domain, domain)

        if self.boundary_condition != None:
            boundary_term = self.get_boundary_term(div, args_with_trial)
            if boundary_term != None:
                integrated_parts = integrated_parts - boundary_term

        self.term = integrated_parts
        debug_print(self.debug, "Transformed Integral: ", self.term, "sympyPprint")
        return integrated_parts
            

    def perform_integration_by_parts_on_gradient(self):
        integral = self.term
        debug_print(self.debug, "Performing integration by parts on gradient integral", self.term, "heading")
        # Get the integrated function
        integral_args = integral.args[0]
        #TODO expression can contain multiple gradient operators
        gradient_function, args_with_trial = get_expression_of_type(grad, integral_args)
        # This expression will be replaced
        grad_test_inner_product = inner(gradient_function, self.test_vector)
        # v must be vector valued
        test_divergence = div(self.test_vector)
        # This expression will be inserted
        trial_div_mult = args_with_trial * test_divergence
        integral_over_domain = integral_args.subs(grad_test_inner_product, trial_div_mult)
        integrated_parts = sympy.Integral(integral_over_domain, domain)

        if self.boundary_condition != None:
            boundary_term = self.get_boundary_term(grad, args_with_trial)
            if boundary_term != None:
                integrated_parts = integrated_parts - boundary_term

        self.term = integrated_parts
        debug_print(self.debug, "Transformed Integral: ", self.term, "sympyPprint")
        return integrated_parts


    def perform_integration_by_parts_on_laplacian(self):
        sympy.Add.precedence_traditional = True
        integral = self.term
        debug_print(self.debug, "Performing integration by parts on laplace integral", self.term, "heading")
        # Get the integrated function
        integral_args = integral.args[0]
        laplacian_function, args_with_trial = get_expression_of_type(Laplacian, integral_args)
        # Define the gradient of the trial function (argument of the laplace function) -> will be vector valued or skalar valued, depending on the use of u or u_vec as Laplacian argument
        trial_gradient = grad(args_with_trial)
        integrated_parts = None
        if self.dim != Dimensions.skalar and self.dim != Dimensions.vector:
            raise Exception("Cannot perform integration by parts on laplacian function - dimension unknown")
       
        # TODO inner products have to be handeled different
        if contains_function_on_surface(inner, self.term):
            # Function look like: inner(Laplacian(u_vec), v-vec) -> Laplacian(u_vec) results in Vector

            existing_inner, inner_args = get_inner(integral_args)
            laplacian_test_inner = inner(laplacian_function, self.test_vector)

            arg_with_laplace = None
            new_args = []
            for arg in inner_args:
                if arg.has(Laplacian):
                    arg_with_laplace = arg
                else:
                    new_args.append(arg)

            test_gradient = grad(self.test_vector)
            #  replace the inner product of inner(Laplacian(u_vec), v_vec)
            inner_func = inner(trial_gradient, test_gradient)
            integral_over_domain = integral_args.subs(existing_inner, inner_func)
            integrated_parts = sympy.Integral(-integral_over_domain, domain)

        if self.dim == Dimensions.skalar:
            # Function look like: Laplacian(u) * v -> Laplacian(u) results in skalar
            test_gradient = grad(self.test)
            #  replace the multiplication of laplace(u) * v
            laplacian_test_mult = laplacian_function * self.test
            inner_func = inner(trial_gradient, test_gradient)
            integral_over_domain = integral_args.subs(laplacian_test_mult, inner_func)
            integrated_parts = sympy.Integral(integral_over_domain, domain)

        if self.boundary_condition != None:
            boundary_term = self.get_boundary_term(Laplacian, args_with_trial)
            if boundary_term != None:
                integrated_parts = integrated_parts - boundary_term

        self.term = integrated_parts
        debug_print(self.debug, "Transformed Integral: ", self.term, "sympyPprint")
        return integrated_parts

    def get_boundary_term(self, operation, args_with_trial: sympy.Expr):
        integrated_surface = None
        if self.boundary_condition != None:
            if self.boundary_condition == Boundaries.neumann and self.boundary_function == None:
                raise Exception("Need to provide a boundary function symbol to use neumann boundaries")
            
            if self.boundary_condition == Boundaries.neumann:
                included_symbols = args_with_trial.free_symbols

                if operation == div:
                    included_trials = [sym for sym in included_symbols if sym in self.trial_vector]
                    if len(included_trials) > 0:
                        surface_term = args_with_trial.subs(included_trials[0], self.boundary_function["div"]) * self.test
                        integrated_surface = sympy.Integral(surface_term, surface)

                elif operation == grad:
                    included_trials = [sym for sym in included_symbols if sym in self.trial]
                    print("REPLACE")
                    if len(included_trials) > 0:
                        surface_term = inner(args_with_trial.subs(included_trials[0], self.boundary_function["grad"]), self.test_vector)
                        integrated_surface = sympy.Integral(surface_term, surface)


                elif operation == curl:
                    included_trials = [sym for sym in included_symbols if sym in self.trial_vector]
                    if len(included_trials) > 0:
                        surface_term = inner(args_with_trial.subs(included_trials[0], self.boundary_function["curl"]), self.test_vector)
                        integrated_surface = sympy.Integral(surface_term, surface)


                elif operation == Laplacian:
                    included_trials = [sym for sym in included_symbols if sym in self.trial]
                    if len(included_trials) > 0:
                        surface_term = args_with_trial.subs(included_trials[0], self.boundary_function["laplacian"]) * self.test_vector
                        integrated_surface = sympy.Integral(surface_term, surface)

        return integrated_surface


    '''
    UFL-Conversion
    '''
    def convert_integral_to_ufl_string(self):
        integral_term = self.term
        if self.term == 0:
            return "0"
        integral_atom = list(integral_term.atoms(sympy.Integral))[0]

        integral_args = integral_atom.args[0]
        term_without_integral = integral_term.subs(integral_atom, integral_args)
        integral_domain = integral_atom.args[1][0]
        replaced_as_string = str(term_without_integral)
        # Every part of the term will be put inside the integral here. Should be mathematically correct, as it can only be a factor in front of the integral, which will be pulled in
        if integral_args == 0:
            return "0"
        if integral_domain == domain:
            string_as_integral_string = "(" + replaced_as_string + ") * dx"
        elif integral_domain == surface:
            string_as_integral_string = "(" + replaced_as_string + ") * ds"
        else: 
            string_as_integral_string = replaced_as_string
        self.ufl_string = string_as_integral_string
        return string_as_integral_string

    '''
    Validation
    '''
    def check_linearity(self, checkFunction):
        exponential = self.term.atoms(sympy.Pow)
        if len(exponential) > 0:    
            for exponential_arg in exponential:
                diff_operator_in_exponential = exponential_arg.atoms(checkFunction)
                if len(diff_operator_in_exponential) > 0:
                    debug_print(True, "Exponent in differential Operator - not implemented", self.term)
                    raise Exception("Exponent in differential operators are not allowed")
                

    def is_nonlinear(self):
        typed_expressions = get_expression_types(expression=self.term)
        linearity_level = 0
        for typed_e in typed_expressions:
            if typed_e["type"] == "mul":
                included_atoms = typed_e["expression"].atoms()
                included_trial = set(included_atoms) & set(self.trial)
                included_trial_vector = set(included_atoms) & set(self.trial_vector)
                if len(included_trial_vector) > 0 or len(included_trial) > 0:
                    linearity_level = linearity_level + 1

        if linearity_level > 1:
            return True
        else:
            return False
