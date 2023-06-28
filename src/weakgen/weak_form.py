from typing import Optional, List
import sympy
from .scripts.integral.util.boundaries.boundaries import Boundaries, BoundaryFunctions
from .scripts.preprocessing.preprocessing import parse_string_equation
from .scripts.integral.integral import Integral
from .util.util import execute_test_multiplications, execute_integration, execute_integration_by_parts, execute_ufl_conversion, sort_terms



class Weak_form:
    def __init__(self, trial_function_names: Optional[List[str]] = None, test_function_names: Optional[List[str]] = None, vector_trial_fuction_names: Optional[List[str]] = None, vector_test_function_names: Optional[List[str]] = None, sympy_equation: Optional[sympy.Eq] = None, string_equation: Optional[str] = None, variables: Optional[List[str]] = None, variable_vectors: Optional[List[str]] = None, boundary_condition: Optional[Boundaries] = Boundaries.dirichlet, boundary_function: Optional[dict[str,str]] = None, debug: Optional[bool] = True):
        '''
        See [GitHub](https://github.com/Denn1sMay/weak) for further Details 
        ## Example Usage

        ```
        from weakgen import Weak_form
        from ufl import inner, grad, div, curl, div, ds, dx
        
        weak_form_object = Weak_form(trial_function_names=["u"], test_function_names=["v"], string_equation="Laplacian(u) = f")
        weak_form_lhs_string, weak_form_rhs_string = weak_form_object.solve()
        # Result:
        # weak_form_lhs_string: (-inner(grad(u), grad(v))) * dx
        # weak_form_rhs_string: (f*v) * dx        a_as_dolfin_expr = eval(weak_form_lhs_string)
        L_as_dolfin_expr = eval(weak_form_rhs_string)
        ```

        The ufl operators have to be imported as in the example. Otherwise the eval() function wont be able to map the string functions to the ufl-implementation of the functions.


        ### Accessible differential operators
        - Laplacian (take the laplacian of an expression)
        - grad (take the gradient of an expression)
        - div (take the divergence of an expression)
        - curl (take the curl of an exression)

        ### Applicable Boundary Conditions
        You can specify a boundary condition by passing a condition to the `boundary_condition` parameter. If you need a Neumann boundary condition, you must provide a string literal for the `boundary_function` parameter. The available options are:
        - Boundaries.dirichlet (default): Surface integrals will vanish.
        - Boundaries.neumann: Requires the `boundary_function` parameter. Applies a Neumann boundary condition.
       
        ### Input
        To use the package, provide the strong form of the equation as a string to the `equation_string` parameter. The equation can be scalar- or vector-valued, as it will be converted to a scalar weak form.
        You also need to provide the string literals of the trial function(s) in a list to the `trial_function_names` or `vector_trial_function_names` parameter, depending on their dimension. Similarly, provide the string literals of the test function(s) defined in your program's scope to the `test_function_names` or `vector_test_function_names` parameter, based on the dimension of your equation.
        If you encounter any difficulties during the conversion process, you can set the `debug=True` parameter to receive hints on where the conversion failed.

        ### What does it do
        The Weak Form Equation Generator attempts to parse your string equation into a sympy equation and separate its terms. When you call solve() on the returned object, the following steps will be executed:

        1. Multiply the equation with the test function.
        2. Integrate the equation over the domain.
        3. Apply integration by parts to terms containing differential operators.
        4. Convert the equation to a UFL-syntaxed string.

        The two resulting string values represent the left-hand side (LHS) and the right-hand side (RHS) of the weak form equation. You can use the built-in eval() function in Python to parse the string equation into the variables present in your program's scope.

        '''
        self.trial = [sympy.Symbol(tr) for tr in trial_function_names] if trial_function_names != None else []
        self.test = [sympy.Symbol(te) for te in test_function_names] if test_function_names != None else []
        self.trial_vector = [sympy.Symbol(vtr) for vtr in vector_trial_fuction_names] if vector_trial_fuction_names != None else []

        self.test_vector = [sympy.Symbol(vte) for vte in vector_test_function_names] if vector_test_function_names != None else []
        self.variables = [sympy.Symbol(va) for va in variables] if variables != None else []
        self.variable_vectors = [sympy.Symbol(va) for va in variable_vectors] if variable_vectors != None else []
        self.boundary_func = boundary_function
        self.surface = sympy.Symbol("surface") if boundary_function != None else None
        self.equation = parse_string_equation(string_equation) if string_equation != None else sympy_equation
        self.boundary = boundary_condition
        self.debug = debug
        self.lhs_terms = []
        self.rhs_terms = []

        self.make_sorted_terms()
        self.assume_dimensions()
        self.verify_dimensions()


    def solve(self):
        self.multiply_with_test_function()
        self.integrate_over_domain()
        self.integraty_by_parts()
        self.convert_to_ufl_string()
        self.debug_print("Weak Form Solution:", "heading")
        self.debug_print("", "sympyPprint")
        print("UFL formatted weak form:")
        print(str(self.lhs_ufl_string) + " = " + str(self.rhs_ufl_string))
        return self.lhs_ufl_string, self.rhs_ufl_string

    def make_sorted_terms(self):
        '''
            sort the equation (trial function to lhs) and make arguments (Summands)
        '''
        self.debug_print("Creating equation arguments", "heading")
        lhs_args = sympy.Add.make_args(self.equation.lhs)
        rhs_args = sympy.Add.make_args(self.equation.rhs)
        new_lhs_terms = []
        new_rhs_terms = []
        sorted_lhs_from_lhs_terms, sorted_rhs_from_lhs_terms = sort_terms(lhs_args, "lhs", trial=self.trial, test=self.test, trial_vector=self.trial_vector, test_vector=self.test_vector, variables=self.variables, variable_vectors=self.variable_vectors, boundary=self.boundary, boundary_func=self.boundary_func, debug=self.debug)
        sorted_lhs_from_rhs_terms, sorted_rhs_from_rhs_terms = sort_terms(rhs_args, "rhs", trial=self.trial, test=self.test, trial_vector=self.trial_vector, test_vector=self.test_vector, variables=self.variables, variable_vectors=self.variable_vectors, boundary=self.boundary, boundary_func=self.boundary_func, debug=self.debug)
        for lhs_term in sorted_lhs_from_rhs_terms + sorted_lhs_from_lhs_terms:
            new_lhs_terms.append(lhs_term)
        for rhs_term in sorted_rhs_from_lhs_terms + sorted_rhs_from_rhs_terms:
            new_rhs_terms.append(rhs_term)

        self.lhs_terms = new_lhs_terms
        self.rhs_terms = new_rhs_terms



    def update_equation(self):
        '''
        Update the equation using modified Integral-Objects
        '''
        lhs = 0
        rhs = 0
        for term in self.lhs_terms:
            lhs = sympy.Add(lhs, term.term)
        for term in self.rhs_terms:
            rhs = sympy.Add(rhs, term.term)
        self.equation = sympy.Eq(lhs, rhs)
        self.debug_print("Updated Equation:", "sympyPprint")
        

    def multiply_with_test_function(self):
        self.debug_print("Multiply with Test Function", "heading")
        self.lhs_terms = execute_test_multiplications(self.lhs_terms)
        self.rhs_terms = execute_test_multiplications(self.rhs_terms)
        self.update_equation()

    def integrate_over_domain(self):
        self.debug_print("Integrate over domain", "heading")
        self.lhs_terms = execute_integration(self.lhs_terms)
        self.rhs_terms = execute_integration(self.rhs_terms)
        self.update_equation()

    def integraty_by_parts(self):
        self.debug_print("Integrate by parts", "heading")
        self.lhs_terms = execute_integration_by_parts(self.lhs_terms)
        self.rhs_terms = execute_integration_by_parts(self.rhs_terms)
        self.update_equation()
        self.make_sorted_terms()
        self.update_equation()

    def convert_to_ufl_string(self):
        self.debug_print("Convert to UFL string", "heading")
        self.lhs_ufl_string = execute_ufl_conversion(self.lhs_terms)
        self.rhs_ufl_string = execute_ufl_conversion(self.rhs_terms)
        self.debug_print("UFL Left Hand Side:")
        self.debug_print(self.lhs_ufl_string)
        self.debug_print("UFL Right Hand Side:")
        self.debug_print(self.rhs_ufl_string)
        self.debug_print("")

    def assume_dimensions(self):
        dimension = None
        all_dims = list(map(lambda integral: integral.dim, self.rhs_terms + self.lhs_terms))
        all_dims_not_none = list(filter(lambda dim: dim != None, all_dims))
        dimension = max(set(all_dims_not_none), key=all_dims_not_none.count)
        '''
        for term in self.lhs_terms + self.rhs_terms:
            dimension = term.dim
            if dimension != None:
                break
        if dimension == None:
            raise Exception("Could not verify dimension of equation")
        '''
        new_lhs_terms = []
        for lhs_term in self.lhs_terms:
            if lhs_term.dim == None:
                lhs_term.dim = dimension
            new_lhs_terms.append(lhs_term)
        new_rhs_terms = []
        for rhs_term in self.rhs_terms:
            if rhs_term.dim == None:
                rhs_term.dim = dimension
            new_rhs_terms.append(rhs_term)
        self.lhs_terms = new_lhs_terms
        self.rhs_terms = new_rhs_terms
            


    def debug_print(self, message: str, format: Optional[str] = "default"):
        if self.debug == True:
            if format == "heading":
                print("--------------------" + message + "----------------------")
            if format == "default":
                print(message)
            if format == "sympyPprint":
                print(message)
                sympy.pprint(self.equation)
                print("")


    def verify_dimensions(self):
        if len(self.lhs_terms) == 0:
            raise Exception("Could not find term with trial function in provided equation")
        first_term = self.lhs_terms[0]
        for term in self.lhs_terms + self.rhs_terms:
            if term.dim != None and term.dim != first_term.dim:
                print("Dimension mismatch in input term")
                print("Expected:")
                print(first_term.term)
                print("Dimension:")
                print(first_term.dim)
                print("")
                print("Got:")
                print(term.term)
                print("Dimension:")
                print(term.dim)
                raise Exception("Cannot resolve dimension mismatch")