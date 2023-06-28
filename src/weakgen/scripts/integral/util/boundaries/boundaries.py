from enum import Enum
from typing import TypedDict



class Boundaries(Enum):
    dirichlet = 1
    neumann = 2
    

class BoundaryFunctions(TypedDict):
    curl: str
    div: str
    grad: str
    laplacian: str