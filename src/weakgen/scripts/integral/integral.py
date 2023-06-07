import sympy
from sympy.vector import Laplacian
from .util.dimensions.dimensions import Dimensions
from .util.boundaries.boundaries import Boundaries
from .util.operators.operators import div, grad, curl, inner
from .util.util import calculate_dimension, get_differential_function, replace_div_grad_with_laplace
from typing import Optional


domain = sympy.Symbol("omega")
surface = sympy.Symbol("surface")

class Integral:
    def __init__(self, term: sympy.Expr, trial: Optional[sympy.Symbol] = None, test: Optional[sympy.Symbol] = None, trial_vector: Optional[sympy.Symbol] = None, test_vector: Optional[sympy.Symbol] = None, boundary_condition: Optional[Boundaries] = None, boundary_function: Optional[sympy.Symbol] = None):
        self.term = replace_div_grad_with_laplace(term)
        self.trial = trial
        self.test = test
        self.boundary_condition = boundary_condition
        self.boundary_function = boundary_function
        self.test_vector = test_vector
        self.trial_vector = trial_vector
        self.boundary_condition = boundary_condition
        self.boundary_function = boundary_function
        self.dim = calculate_dimension(term, trial, trial_vector)

    def multiply_with_test_function(self):
        if self.dim == Dimensions.skalar:
            self.term = self.term * self.test
        elif self.dim == Dimensions.vector:
            # TODO need a vector valued test function here to create inner product of two vectors
            self.term = inner(self.term, self.test_vector)
        else:
            self.term = self.term * self.test

    def integrate_over_domain(self):
        self.term = sympy.Integral(self.term, domain)

    '''
    Integration by Parts
    '''
    def integrate_by_parts(self):
        if self.term.has(div):
            self.check_linearity(div)
            if self.term.has(Laplacian):
                raise Exception("Divergence and Laplace cannot be combined")
            self.perform_integration_by_parts_on_diveregence()
        elif self.term.has(Laplacian):
            self.check_linearity(Laplacian)
            self.perform_integration_by_parts_on_laplacian()
        elif self.term.has(grad):
            self.check_linearity(grad)
            self.perform_integration_by_parts_on_gradient()
        elif self.term.has(curl):
            self.check_linearity(curl)
            self.perform_integration_by_parts_on_curl()
        else:
            print("No differential operator present - skipping integration by parts")
            
    def perform_integration_by_parts_on_curl(self):
        integral = self.term
        print("Performing integration by parts on curl integral")
        sympy.pprint(integral)
        # Get the curl function
        integral_args = integral.args[0]
        #TODO expression can contain multiple curl operators
        curl_function, args_with_trial = get_differential_function(curl, integral_args)
        # This expression will be replaced
        curl_test_inner_product = inner(curl_function, self.test_vector)
        print("curl_test_inner")
        sympy.pprint(curl_test_inner_product)
        # v must be vector valued
        test_curl = curl(self.test_vector)
        # This expression will be inserted
        trial_curl_inner = inner(args_with_trial, test_curl)
        integral_over_domain = integral_args.subs(curl_test_inner_product, trial_curl_inner)
        integrated_parts = sympy.Integral(-integral_over_domain, domain)

        if self.boundary_condition != None:
            if self.boundary_condition == Boundaries.neumann and self.boundary_function == None:
                raise Exception("Need to provide a boundary function symbol to use neumann boundaries")
            if self.boundary_condition == Boundaries.neumann:
                if(args_with_trial.has(curl)):
                    raise Exception("Neumann boundaries for curl terms not yet implemented")
                
        print("Transformed Integral: ")
        sympy.pprint(integrated_parts)
        self.term = integrated_parts
        return integrated_parts
      

    def perform_integration_by_parts_on_diveregence(self):
        integral = self.term
        print("Performing integration by parts on divergence integral")
        sympy.pprint(integral)
        integral_args = integral.args[0]
        divergence_function, args_with_trial = get_differential_function(div, integral_args)
        test_gradient = grad(self.test)
        div_test_mult = divergence_function * self.test
        inner_func = inner(args_with_trial, test_gradient)
        integral_over_domain = integral_args.subs(div_test_mult, inner_func)
        integrated_parts = sympy.Integral(-integral_over_domain, domain)
        if self.boundary_condition != None:
            if self.boundary_condition == Boundaries.neumann and self.boundary_function == None:
                raise Exception("Need to provide a boundary function symbol to use neumann boundaries")
            if self.boundary_condition == Boundaries.neumann:
                if(args_with_trial.has(grad)):
                    nabla_grad_term = min(args_with_trial.atoms(grad))
                    nabla_grad_args = nabla_grad_term.args[0]
                    #TODO mathematically incorrect
                    dn = sympy.Symbol("dn")
                    nabla_grad_args_div_n = nabla_grad_args / dn
                    boundary_and_residual = nabla_grad_args_div_n.subs((self.trial/dn), self.boundary_function)
                    integral_over_boundary = integral_args.subs(div_test_mult, (boundary_and_residual * self.test))
                    integrated_parts = integrated_parts + sympy.Integral(integral_over_boundary, surface)
                else:
                    print("Vector Function spaces are not implemented yet")
                    print("Cannot Add Neumann boundary")
                    raise Exception("Neumann Boundaries for vector valued trial functions are not implemented yet")
        print("Transformed Integral: ")
        sympy.pprint(integrated_parts)
        self.term = integrated_parts
        return integrated_parts
            

    def perform_integration_by_parts_on_gradient(self):
        integral = self.term
        print("Performing integration by parts on gradient integral")
        sympy.pprint(integral)
        # Get the integrated function
        integral_args = integral.args[0]
        #TODO expression can contain multiple gradient operators
        gradient_function, args_with_trial = get_differential_function(grad, integral_args)
        # This expression will be replaced
        grad_test_inner_product = inner(gradient_function, self.test_vector)
        print("grad_test_inner")
        sympy.pprint(grad_test_inner_product)
        # v must be vector valued
        test_divergence = div(self.test_vector)
        # This expression will be inserted
        trial_div_mult = args_with_trial * test_divergence
        integral_over_domain = integral_args.subs(grad_test_inner_product, trial_div_mult)
        integrated_parts = sympy.Integral(-integral_over_domain, domain)

        if self.boundary_condition != None:
            if self.boundary_condition == Boundaries.neumann and self.boundary_function == None:
                raise Exception("Need to provide a boundary function symbol to use neumann boundaries")
            if self.boundary_condition == Boundaries.neumann:
                if(args_with_trial.has(grad)):
                    raise Exception("Neumann boundaries for gradient terms not yet implemented")
                
        print("Transformed Integral: ")
        sympy.pprint(integrated_parts)
        self.term = integrated_parts
        return integrated_parts


    def perform_integration_by_parts_on_laplacian(self):
        sympy.Add.precedence_traditional = True
        integral = self.term
        print("Performing integration by parts on laplace integral")
        sympy.pprint(integral)
        # Get the integrated function
        integral_args = integral.args[0]
        #TODO argument of laplace operator can be a complex expression
        laplacian_function, args_with_trial = get_differential_function(Laplacian, integral_args)
        # Define the gradient of the trial function (argument of the laplace function)
        trial_gradient = grad(args_with_trial)
        # Define the nabla operator on the test function
        test_gradient = grad(self.test)
        # Perform the integration by parts 
        #  -> replace the multiplication of laplace(u) * v
        laplacian_test_mult = laplacian_function * self.test
        inner_func = inner(trial_gradient, test_gradient)
        integral_over_domain = integral_args.subs(laplacian_test_mult, inner_func)
        integrated_parts = sympy.Integral(-integral_over_domain, domain)
        if self.boundary_condition != None:
            if self.boundary_condition == Boundaries.neumann and self.boundary_function == None:
                raise Exception("Need to provide a boundary function symbol to use neumann boundaries")
            if self.boundary_condition == Boundaries.neumann:
                boundary_and_residual = args_with_trial.subs(self.trial, self.boundary_function)
                integral_over_boundary = integral_args.subs(laplacian_test_mult, (boundary_and_residual * self.test))
                integrated_parts = integrated_parts + sympy.Integral(integral_over_boundary, surface)
        print("Transformed Integral: ")
        sympy.pprint(integrated_parts)
        self.term = integrated_parts
        return integrated_parts

    '''
    UFL-Conversion
    '''
    def convert_integral_to_ufl_string(self):
        integral_term = self.term
        integral_atom = list(integral_term.atoms(sympy.Integral))[0]

        integral_args = integral_atom.args[0]
        term_without_integral = integral_term.subs(integral_atom, integral_args)
        integral_domain = integral_atom.args[1][0]
        replaced_as_string = str(term_without_integral)
        # Every part of the term will be put inside the integral here. Should be mathematically correct, as it can only be a factor in front of the integral, which will be pulled in
        if integral_domain == domain:
            string_as_integral_string = "(" + replaced_as_string + ") * dx"
        elif integral_domain == surface:
            string_as_integral_string = "(" + replaced_as_string + ") * ds"
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
                    print("Exponent in differential Operator - not implemented")
                    raise Exception("Exponent in differential operators are not allowed")
