


def execute_test_multiplications(terms: list):
    multiplied_terms = []
    for term in terms:
        term.multiply_with_test_function()
        multiplied_terms.append(term)
    return multiplied_terms

def execute_integration(terms: list):
    integrated_terms = []
    for term in terms:
        term.integrate_over_domain()
        integrated_terms.append(term)
    return integrated_terms

def execute_integration_by_parts(terms: list):
    partially_integrated_terms = []
    for term in terms:
        term.integrate_by_parts()
        partially_integrated_terms.append(term)
    return partially_integrated_terms