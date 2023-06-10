from typing import Optional
import sympy
from .scripts.integral.util.boundaries.boundaries import Boundaries
from .scripts.preprocessing.preprocessing import parse_string_equation
from .scripts.integral.integral import Integral
from .util.util import execute_test_multiplications, execute_integration, execute_integration_by_parts, execute_ufl_conversion, sort_terms


class Weak_form:
    def __init__(self, trial_function_name: str, test_function_name: str, vector_trial_fuction_name: Optional[str] = None, vector_test_function_name: Optional[str] = None, equation: Optional[sympy.Eq] = None, string_equation: Optional[str] = None, boundary_condition: Optional[Boundaries] = None, boundary_function: Optional[str] = None, debug: Optional[bool] = True):
        self.trial = sympy.Symbol(trial_function_name)
        self.test = sympy.Symbol(test_function_name)
        self.boundary = boundary_condition
        self.debug = debug
        self.lhs_terms = []
        self.rhs_terms = []

        if string_equation != None:
            self.equation = parse_string_equation(string_equation)
        else:
            self.equation = equation

        if boundary_function != None:
            self.boundary_func = sympy.Symbol(boundary_function)
            self.surface = sympy.Symbol("surface")
        else:
            self.boundary_func = None

        if vector_trial_fuction_name != None:
            self.trial_vector = sympy.Symbol(vector_trial_fuction_name)
        else:
            self.trial_vector = None

        if vector_test_function_name != None:
            self.test_vector = sympy.Symbol(vector_test_function_name)
        else:
            self.test_vector = None

        self.make_sorted_terms()
        self.assume_dimensions()
        self.verify_dimensions()


    def solve(self):
        self.multiply_with_test_function()
        self.integrate_over_domain()
        self.integraty_by_parts()
        self.convert_to_ufl_string()
        print("UFL formatted weak form:")
        print(str(self.lhs_ufl_string) + " = " + str(self.rhs_ufl_string))

    def make_sorted_terms(self):
        '''
            sort the equation (trial function to lhs) and make arguments (Summands)
        '''
        self.debug_print("Creating equation arguments", "heading")
        lhs_args = sympy.Add.make_args(self.equation.lhs)
        rhs_args = sympy.Add.make_args(self.equation.rhs)
        new_lhs_terms = []
        new_rhs_terms = []
        sorted_lhs_from_lhs_terms, sorted_rhs_from_lhs_terms = sort_terms(lhs_args, "lhs", trial=self.trial, test=self.test, trial_vector=self.trial_vector, test_vector=self.test_vector, boundary=self.boundary, boundary_func=self.boundary_func, debug=self.debug)
        sorted_lhs_from_rhs_terms, sorted_rhs_from_rhs_terms = sort_terms(rhs_args, "rhs", trial=self.trial, test=self.test, trial_vector=self.trial_vector, test_vector=self.test_vector, boundary=self.boundary, boundary_func=self.boundary_func, debug=self.debug)
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
        for term in self.lhs_terms + self.rhs_terms:
            dimension = term.dim
            if dimension != None:
                break
        if dimension == None:
            raise Exception("Could not verify dimension of equation")
        new_lhs_terns = []
        for lhs_term in self.lhs_terms:
            if lhs_term.dim == None:
                lhs_term.dim = dimension
            new_lhs_terns.append(lhs_term)
        new_rhs_terns = []
        for rhs_term in self.rhs_terms:
            if rhs_term.dim == None:
                rhs_term.dim = dimension
            new_rhs_terns.append(rhs_term)
        self.lhs_terms = new_lhs_terns
        self.rhs_terms = new_rhs_terns
            


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