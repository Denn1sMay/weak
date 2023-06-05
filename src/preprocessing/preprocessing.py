import sympy
from ..integral.util.operators.operators import div
from sympy.vector import Laplacian

def use_sympy_laplace_operator(sympy_term):
    undefined_laplacian_function = sympy.Function("Laplacian")
    lap_atoms = sympy_term.atoms(undefined_laplacian_function)
    new_term = sympy_term
    for lap_atom in lap_atoms:
        laplacian_args = lap_atom.args
        print(type(laplacian_args[0]))
        lap_operator = Laplacian(laplacian_args[0])
        new_term = new_term.subs(lap_atom, lap_operator)
    return new_term


def parse_string_equation(string_equation: str):
    print(string_equation)
    custom_dict = {"div": div}
    equation_sides = string_equation.split("=")
    if len(equation_sides) > 2:
        raise ValueError("Equation contained more than one '='")

    if len(equation_sides) == 2:
        lhs_parsed = sympy.parse_expr(equation_sides[0], local_dict=custom_dict, evaluate=False)
        rhs_parsed = sympy.parse_expr(equation_sides[1], local_dict=custom_dict, evaluate=False)
        print("lhs:")
        sympy.pprint(lhs_parsed)
        print("rhs:")
        sympy.pprint(rhs_parsed)
        lhs_with_operators = use_sympy_laplace_operator(lhs_parsed)
        rhs_with_operators = use_sympy_laplace_operator(rhs_parsed)
    else:
        lhs_parsed = sympy.parse_expr(equation_sides[0], local_dict=custom_dict, evaluate=False)
        print("lhs:")
        sympy.pprint(lhs_parsed)
        rhs_with_operators = sympy.parse_expr("0", evaluate=False)
        lhs_with_operators = use_sympy_laplace_operator(lhs_parsed)

    parsed_equation = sympy.Eq(lhs_with_operators, rhs_with_operators)
    print("Result with sympy operators:")
    sympy.pprint(parsed_equation)
    return parsed_equation

