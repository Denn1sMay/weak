from typing import Optional
import sympy
from src.integral.util.boundaries.boundaries import Boundaries
from src.preprocessing.preprocessing import parse_string_equation
from src.integral.integral import Integral
from src.util.util import execute_test_multiplications, execute_integration, execute_integration_by_parts

class Weak_form:
    def __init__(self, trial_function_name: str, test_function_name: str, vector_trial_fuction_name: Optional[str] = None, vector_test_function_name: Optional[str] = None, equation: Optional[sympy.Eq] = None, string_equation: Optional[str] = None, boundary_condition: Optional[Boundaries] = None, boundary_function: Optional[str] = None):
        self.trial = sympy.Symbol(trial_function_name)
        self.test = sympy.Symbol(test_function_name)
        self.boundary = boundary_condition
        self.lhs_terms = []
        self.rhs_terms = []
        if string_equation != None:
            self.equation = parse_string_equation(string_equation)
        else:
            self.equation = equation
        if boundary_function != None:
            self.boundary_func = sympy.Symbol(boundary_function)
            self.surface = sympy.Symbol("surface")
        if vector_trial_fuction_name != None:
            self.trial_vector = sympy.Symbol(vector_trial_fuction_name)
        if vector_test_function_name != None:
            self.test_vector = sympy.Symbol(vector_test_function_name)
        self.make_sorted_terms()
        self.verify_dimensions()


    def make_sorted_terms(self):
        '''
            sort the equation (trial function to lhs) and make arguments (Summands)
        '''
        lhs_sorted = 0
        rhs_sorted = 0
        lhs_args = sympy.Add.make_args(self.equation.lhs)
        rhs_args = sympy.Add.make_args(self.equation.rhs)
        new_lhs_terms = []
        new_rhs_terms = []
        for arg in lhs_args + rhs_args:
            if hasattr(self, "trial") and hasattr(self, "trial_vector"):
                print("both")
                if arg.has(self.trial) or arg.has(self.trial_vector):
                    new_lhs_terms.append(Integral(arg, trial=self.trial, test=self.test, trial_vector=self.trial_vector, test_vector=self.test_vector, boundary_condition=self.boundary, boundary_function=self.boundary_func))
                else:
                    new_rhs_terms.append(Integral(arg, trial=self.trial, test=self.test, trial_vector=self.trial_vector, test_vector=self.test_vector, boundary_condition=self.boundary, boundary_function=self.boundary_func))
            elif hasattr(self, "trial") and not hasattr(self, "trial_vector"):
                print("trial")

                if arg.has(self.trial):
                    new_lhs_terms.append(Integral(arg, trial=self.trial, test=self.test, boundary_condition=self.boundary, boundary_function=self.boundary_func))
                else:
                    new_rhs_terms.append(Integral(arg, trial=self.trial, test=self.test, boundary_condition=self.boundary, boundary_function=self.boundary_func))
            elif hasattr(self, "trial_vector") and not hasattr(self, "trial"):
                print("vector")
                if arg.has(self.trial_vector):
                    new_lhs_terms.append(Integral(arg, trial_vector=self.trial_vector, test_vector=self.test_vector, boundary_condition=self.boundary, boundary_function=self.boundary_func))
                else:
                    new_rhs_terms.append(Integral(arg, trial_vector=self.trial_vector, test_vector=self.test_vector, boundary_condition=self.boundary, boundary_function=self.boundary_func))
            else:
                raise Exception("Need to provide string literals of trial- and test function(s)")
            self.lhs_terms = new_lhs_terms
            self.rhs_terms = new_rhs_terms



    def update_equation(self):
        lhs = 0
        rhs = 0
        for term in self.lhs_terms:
            lhs = sympy.Add(lhs, term.term)
        for term in self.rhs_terms:
            rhs = sympy.Add(rhs, term.term)
        self.equation = sympy.Eq(lhs, rhs)
        

    def multiply_with_test_function(self):
        self.lhs_terms = execute_test_multiplications(self.lhs_terms)
        self.rhs_terms = execute_test_multiplications(self.rhs_terms)

        self.update_equation()

    def integrate_over_domain(self):
        self.lhs_terms = execute_integration(self.lhs_terms)
        self.rhs_terms = execute_integration(self.rhs_terms)
        self.update_equation()

    def integraty_by_parts(self):
        self.lhs_terms = execute_integration_by_parts(self.lhs_terms)
        self.rhs_terms = execute_integration_by_parts(self.rhs_terms)
        self.update_equation()
        self.make_sorted_terms()
        self.update_equation()


    def verify_dimensions(self):
        first_term = self.lhs_terms[0]
        for term in self.lhs_terms + self.rhs_terms:
            if term.dim != first_term.dim:
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
                raise Exception()