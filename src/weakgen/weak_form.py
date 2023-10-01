from typing import Optional, List
import sympy
from .scripts.integral.util.boundaries.boundaries import Boundaries, BoundaryFunctions
from .scripts.preprocessing.preprocessing import parse_string_equation
from .scripts.integral.integral import Integral
from .util.util import get_executable_string, get_sympy_symbols, execute_test_multiplications, execute_integration, execute_integration_by_parts, execute_ufl_conversion, sort_terms



class Weak_form:
    def __init__(self, 
                 functions: dict,
                 mesh,
                 sympy_equation: Optional[sympy.Eq] = None, 
                 string_equation: Optional[str] = None, 
                 variables: Optional[List[str]] = None, 
                 variable_vectors: Optional[List[str]] = None, 
                 boundary_condition: Optional[Boundaries] = Boundaries.dirichlet, 
                 boundary_function: Optional[dict[str,str]] = None, 
                 debug: Optional[bool] = False):
        '''
        See [GitHub](https://github.com/Denn1sMay/weak) for further Details 
        ## Example Usage

        ```python
        from mpi4py import MPI
        from dolfinx import mesh, fem

        from weakgen import Weak_form, Boundaries

        domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)

        u_dict = {
            "u": {
                "order": 1,
                "dim": "scalar",
                "spaceName": "Myspace"
            }
        }
        f = fem.Constant(domain, ScalarType(-6))
        pde = "Laplacian(u) = f"

        weak_form_object = Weak_form(functions=u_dict, mesh="domain", string_equation=pde, boundary_condition=Boundaries.dirichlet)

        # lhs and rhs of equation as strings
        # commands contains a python-executable script
        a_generated_string, L_generated_string, commands = weak_form_object.solve()
        # Declares Finite Elements and FunctionSpaces, TrialFunctions and Testfunctions
        # Declares lhs of the Equation as "a" and rhs as "L"
        exec(commands)

        # After defining your Boundary conditions "bc"
        # The variables a and L can be used like that:

        problem = fem.petsc.LinearProblem(a, L, bcs=[bc])
        uh = problem.solve()
        ```

        ___

        ### Input
        To use the package, provide the strong form of the equation as a string to the `equation_string` parameter. The equation can be scalar- or vector-valued, as it will be converted to a scalar weak form.
        You also need to provide the unknown functions of your equation. Define them as a dict like:
        ```python
        {dict_name} = {
            "{variable_name}": {
                "order": {number},
                "dim": {"scalar" | "vector" | "matrix"},
                "spaceName": {optional: string}
            }
        }

        # Example:
        variables = {
            "u": {
                "order": 2,
                "dim": "vector",
            },
            "p": {
                "order": 1,
                "dim": "scalar",
                "spaceName": "my_functionspace_name"
            }
        }
        ```
        You have to provide the polynomial order to the "order"-key and the dimension of your function to the "dim"-key. The "dim" "matrix" corresponds to a helper dimension. It is assumed that you apply a function to a defined TrialFunction, which results in a matrix. The matrix-varialbe will not be initializes inside the "commands"-script!
        Optionally, you can pass the desired name of the Functionspace for each variable. 

        Pass the variable name of your defined mesh to the "mesh"-parameter.

        All Functions and Functionspaces will be initialized by calling the "exec()"-Function with the "commands"-variable, which is returned by the "solve"- method of your Weak_form-object.

        If you want to initilize all functions and spaces on your own, you can use pythons "eval()"-function on the received "a_generated_string" and "L_generated_string"-variables.

        The name of the (mixed) function space defaults to "V", if not specified otherwise. Sub function spaces will be accessible with "V_1", "V_2",... if not specified otherwise.
        The testfunctions will be named like the unknown function with the postfix "_test". The example would produce "u_test" and "p_test".

        If you encounter any difficulties during the conversion process, you can set the `debug=True` parameter to receive hints on where the conversion failed.
        ___
        ### Accessible differential operators

        The package supports the following differential operators:

        - Laplacian (take the laplacian of an expression)
        - grad (take the gradient of an expression)
        - div (take the divergence of an expression)
        - curl (take the curl of an exression)

        ### Applicable Boundary Conditions
        You can specify a boundary condition by passing a condition to the `boundary_condition` parameter. If you need a Neumann boundary condition, you must provide a string literal for the `boundary_function` parameter. The available options are:
        - Boundaries.dirichlet (default): Surface integrals will vanish.
        - Boundaries.neumann: Requires the `boundary_function` parameter. Applies a Neumann boundary condition.

        ___

        ### Troubleshooting
        If you encounter errors, try to pass the names of constant variables to the "variables" or "variable_vectors"-parameter, respectively. This will help to determine the dimension of single expressions.

        ___

        ### What does it do
        The Weak Form Equation Generator attempts to parse your string equation into a sympy equation and separate its terms. When you call solve() on the returned object, the following steps will be executed:

        1. Multiply the equation with the test function.
        2. Integrate the equation over the domain.
        3. Apply integration by parts to terms containing differential operators.
        4. Convert the equation to a UFL-syntaxed string.

        The two resulting string values represent the left-hand side (LHS) and the right-hand side (RHS) of the weak form equation. You can use the built-in eval() function in Python to parse the string equation into the variables present in your program's scope.



        [GitHub](https://github.com/Denn1sMay/weak)

        '''
        self.functions = functions
        self.trial, self.trial_vector, self.trial_tensor, self.test, self.test_vector, self.skalar_dict, self.vector_dict, self.tensor_dict = get_sympy_symbols(functions)
        self.mesh = mesh
        #self.trial = [sympy.Symbol(tr) for tr in trial_function_names] if trial_function_names != None else []
        #self.test = [sympy.Symbol(te) for te in test_function_names] if test_function_names != None else []
        #self.trial_vector = [sympy.Symbol(vtr) for vtr in vector_trial_fuction_names] if vector_trial_fuction_names != None else []
        #self.trial_tensor = [sympy.Symbol(ttr) for ttr in tensor_trial_function_names] if tensor_trial_function_names != None else []
        #self.test_vector = [sympy.Symbol(vte) for vte in vector_test_function_names] if vector_test_function_names != None else []
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
        commands = get_executable_string(self.functions, self.mesh, self.lhs_ufl_string, self.rhs_ufl_string)
        final_string_eq = str(self.lhs_ufl_string) + " = " + str(self.rhs_ufl_string)
        if(self.debug):
            print("Executable:")
            print(commands)
        self.debug_print("Weak Form Solution:", "heading")
        self.debug_print("", "sympyPprint")
        print("UFL formatted weak form:")
        print(final_string_eq)
        print("_" * len(final_string_eq))
        return self.lhs_ufl_string, self.rhs_ufl_string, commands

    def make_sorted_terms(self):
        '''
            sort the equation (trial function to lhs) and make arguments (Summands)
        '''
        self.debug_print("Creating equation arguments", "heading")
        lhs_args = sympy.Add.make_args(self.equation.lhs)
        rhs_args = sympy.Add.make_args(self.equation.rhs)
        new_lhs_terms = []
        new_rhs_terms = []
        sorted_lhs_from_lhs_terms, sorted_rhs_from_lhs_terms = sort_terms(lhs_args, "lhs", trial=self.trial, test=self.test, trial_vector=self.trial_vector, test_vector=self.test_vector, trial_tensor=self.trial_tensor, variables=self.variables, variable_vectors=self.variable_vectors, boundary=self.boundary, boundary_func=self.boundary_func, debug=self.debug)
        sorted_lhs_from_rhs_terms, sorted_rhs_from_rhs_terms = sort_terms(rhs_args, "rhs", trial=self.trial, test=self.test, trial_vector=self.trial_vector, test_vector=self.test_vector, trial_tensor=self.trial_tensor, variables=self.variables, variable_vectors=self.variable_vectors, boundary=self.boundary, boundary_func=self.boundary_func, debug=self.debug)
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