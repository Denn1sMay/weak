import sympy
from ..integral.util.operators.operators import div, grad, curl
from sympy.vector import Laplacian

__operator_types = [div, grad, curl, Laplacian]

def use_sympy_laplace_operator(sympy_term):
    undefined_laplacian_function = sympy.Function("Laplacian")
    lap_atoms = sympy_term.atoms(undefined_laplacian_function)
    new_term = sympy_term
    for lap_atom in lap_atoms:
        laplacian_args = lap_atom.args
        lap_operator = Laplacian(laplacian_args[0])
        new_term = new_term.subs(lap_atom, lap_operator)
    return new_term

def expand_with_operators(expression: sympy.Expr):
    expanded_expr = expression.expand()
    for operator in __operator_types:
        if expanded_expr.has(operator):
            operator_atoms = expanded_expr.atoms(operator)
            for operator_atom in operator_atoms:
                operator_args = operator_atom.args[0]
                operator_summands = sympy.Add.make_args(operator_args)
                as_single_operators = [operator(summand) for summand in operator_summands]
                as_addition = sympy.Add(*as_single_operators)
                expanded_expr = expanded_expr.subs(operator_atom, as_addition)
    return expanded_expr.expand()
    

def parse_string_equation(string_equation: str):
    print("Input Equation:")
    print(string_equation)
    custom_dict = {"div": div}
    equation_sides = string_equation.split("=")
    if len(equation_sides) > 2:
        raise ValueError("Equation contained more than one '='")

    if len(equation_sides) == 2:
        lhs_parsed = sympy.parse_expr(equation_sides[0], local_dict=custom_dict, evaluate=False)
        rhs_parsed = sympy.parse_expr(equation_sides[1], local_dict=custom_dict, evaluate=False)
        lhs_with_operators = use_sympy_laplace_operator(lhs_parsed)
        rhs_with_operators = use_sympy_laplace_operator(rhs_parsed)
    else:
        lhs_parsed = sympy.parse_expr(equation_sides[0], local_dict=custom_dict, evaluate=False)
        rhs_with_operators = sympy.parse_expr("0", evaluate=False)
        lhs_with_operators = use_sympy_laplace_operator(lhs_parsed)

    expanded_lhs = expand_with_operators(lhs_with_operators)
    expanded_rhs = expand_with_operators(rhs_with_operators)
    parsed_equation = sympy.Eq(expanded_lhs, expanded_rhs)
    print("Parsed sympy equation")
    sympy.pprint(parsed_equation)
    print("")
    return parsed_equation

