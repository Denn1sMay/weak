import sympy
from sympy.vector import Laplacian
from .util.dimensions.dimensions import Dimensions
from .util.boundaries.boundaries import Boundaries
from .util.operators.operators import div, grad, rot, inner
from .util.util import calculate_dimension
from typing import Optional


domain = sympy.Symbol("omega")

class Integral:

    def __init__(self, term: sympy.Expr, trial: Optional[sympy.Symbol] = None, test: Optional[sympy.Symbol] = None, trial_vector: Optional[sympy.Symbol] = None, test_vector: Optional[sympy.Symbol] = None, boundary_condition: Optional[Boundaries] = None, boundary_function: Optional[sympy.Symbol] = None):
        self.term = term
        self.trial = trial
        self.test = test
        self.surface = sympy.Symbol("surface")
        self.boundary_condition = boundary_condition
        self.boundary_function = boundary_function
        self.test_vector = test_vector
        self.trial_vector = trial_vector
        self.boundary_condition = boundary_condition
        self.boundary_function = boundary_function
        self.dim = calculate_dimension(term, trial, test)

    def multiply_with_test_function(self):
        if self.dim == Dimensions.skalar:
            self.term = self.term * self.test
        elif self.dim == Dimensions.vector:
            # TODO need a vector valued test function here to create inner product of two vectors
            self.term = inner(self.term, self.test_vector)

    def integrate_over_domain(self):
        self.term = sympy.Integral(self.term, domain)


    def integrate_by_parts(self):
        if self.term.has(div):
            self.check_linearity(div)
            if self.term.has(Laplacian):
                print("Divergenze Operator and Laplace Operator found in Integral - cannot calulate")
                raise Exception("Divergence and Laplace cannot be combined")
            print("Divergence found in current integral")
            self.perform_integration_by_parts_on_diveregence()
        elif self.term.has(Laplacian):
            self.check_linearity(Laplacian)
            print("Laplacian found in current integral:")
            self.perform_integration_by_parts_on_laplacian()
        elif self.term.has(grad):
            self.check_linearity(grad)
            print("Gradient found in current integral")
            self.perform_integration_by_parts_on_gradient()
        else:
            print("No differential operator present - skipping integration by parts")
            



    def perform_integration_by_parts_on_diveregence(self):
        sympy.Add.precedence_traditional = True
        integral = self.term
        sympy.pprint(integral)
        # Get the integrated function
        integral_args = integral.args[0]

        # Get the divergence part of it
        a = integral_args.atoms(div)
        #TODO expression can contain multiple divergence operators
        divergence_function = min(a)
        divergence_args = divergence_function.args
        args_with_trial = divergence_args[0]

        nabla_test_function_v = grad(self.test)

        div_test_mult = divergence_function * self.test
        inner_func = inner(args_with_trial, nabla_test_function_v)

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
                    integrated_parts = integrated_parts + sympy.Integral(integral_over_boundary, self.surface)
                else:
                    print("Vector Function spaces are not implemented yet")
                    print("Cannot Add Neumann boundary")
                    #raise Exception("Divergence Operator can only be applied to vector values")
        print("Transformed Integral: ")
        sympy.pprint(integrated_parts)
        self.term = integrated_parts
        return integrated_parts
            

    def perform_integration_by_parts_on_gradient(self):
        sympy.Add.precedence_traditional = True
        integral = self.term
        print("Performing integration by parts on current integral")
        sympy.pprint(integral)
        # Get the integrated function
        integral_args = integral.args[0]
        domain = integral.args[1]

        # Get the gradient part of it
        a = integral_args.atoms(grad)
        #TODO expression can contain multiple gradient operators
        gradient_function = min(a)
        gradient_args = gradient_function.args
        args_with_trial = gradient_args[0]
        # This expression will be replaced
        grad_test_inner_product = inner(gradient_function, self.test_vector)
        print("grad_test_inner")
        sympy.pprint(grad_test_inner_product)
        # v must be vector valued
        divergence_test_function_v = div(self.test_vector)
        # This expression will be inserted
        trial_div_mult = args_with_trial * divergence_test_function_v
        integral_over_domain = integral_args.subs(grad_test_inner_product, trial_div_mult)
        integrated_parts = sympy.Integral(-integral_over_domain, domain)

        if self.boundary_condition != None:
            if self.boundary_condition == Boundaries.neumann and self.boundary_function == None:
                raise Exception("Need to provide a boundary function symbol to use neumann boundaries")
            if self.boundary_condition == Boundaries.neumann:
                if(args_with_trial.has(div)):
                    raise Exception("Neumann boundaries for gradient terms not yet implemented")
                
        print("Transformed Integral: ")
        sympy.pprint(integrated_parts)
        self.term = integrated_parts
        return integrated_parts


    def perform_integration_by_parts_on_laplacian(self):
        sympy.Add.precedence_traditional = True
        integral = self.term
        print("Performing integration by parts on current integral")
        sympy.pprint(integral)
        # Get the integrated function
        integral_args = integral.args[0]
        domain = integral.args[1]

        # Get the laplacian part of it
        a = integral_args.atoms(Laplacian)
        #TODO equation can contain multiple laplace operators
        laplacian_function = min(a)

        # Get the trial function (arguments/ parameters of the laplacian function)
        laplacian_args = laplacian_function.args
        args_with_trial = laplacian_args[0]
        print("Argument of Laplace Operator: ")
        #TODO argument of laplace operator can be a complex expression
        print(args_with_trial)

        # Define the gradient of the trial function (argument of the laplace function)
        nabla_grad = sympy.Function("grad")

        nabla_trial_function_u = nabla_grad(args_with_trial)
        # Define the nabla operator on the test function
        nabla_test_function_v = nabla_grad(self.test)

        # Perform the integration by parts 
        #  -> replace the multiplication of laplace(u) * v
        laplacian_test_mult = laplacian_function * self.test
        inner_func = sympy.Function("inner")(nabla_trial_function_u, nabla_test_function_v)
        integral_over_domain = integral_args.subs(laplacian_test_mult, inner_func)
        integrated_parts = sympy.Integral(-integral_over_domain, domain)
        if self.boundary_condition != None:
            if self.boundary_condition == Boundaries.neumann and self.boundary_function == None:
                raise Exception("Need to provide a boundary function symbol to use neumann boundaries")
            if self.boundary_condition == Boundaries.neumann:
                boundary_and_residual = args_with_trial.subs(self.trial, self.boundary_function)
                integral_over_boundary = integral_args.subs(laplacian_test_mult, (boundary_and_residual * self.test))
                integrated_parts = integrated_parts + sympy.Integral(integral_over_boundary, self.surface)


        print("")
        print("Transformed Integral: ")
        sympy.pprint(integrated_parts)
        self.term = integrated_parts
        return integrated_parts

    def check_linearity(self, checkFunction):
        exponential = self.term.atoms(sympy.Pow)
        if len(exponential) > 0:    
            for exponential_arg in exponential:
                diff_operator_in_exponential = exponential_arg.atoms(checkFunction)
                if len(diff_operator_in_exponential) > 0:
                    print("Exponent in differential Operator - not implemented")
                    raise Exception("Exponent in differential operators are not allowed")
