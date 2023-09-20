from mpi4py import MPI
from dolfinx import mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)

from dolfinx.fem import FunctionSpace

from ..weakgen.weak_form import Weak_form
from ..weakgen.scripts.integral.util.boundaries.boundaries import Boundaries, BoundaryFunctions
from petsc4py.PETSc import ScalarType
from dolfinx import fem

f = fem.Constant(domain, ScalarType(-6))
phi = 3
u_dict = {
    "u": {
        "order": 1,
        "dim": "scalar",
        "spaceName": "Myspace"
    }
}
pde = "phi * Laplacian(u) = f"

weak_form_object = Weak_form(functions=u_dict, mesh="domain", string_equation=pde, boundary_condition=Boundaries.dirichlet)

a_generated_string, L_generated_string, commands = weak_form_object.solve()
exec(commands)


uD = fem.Function(Myspace)
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

import numpy
# Create facet to cell connectivity required to determine boundary facets
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

boundary_dofs = fem.locate_dofs_topological(Myspace, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)

import ufl
#u = ufl.TrialFunction(V)
#v = ufl.TestFunction(V)



#a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
#L = f * v * ufl.dx


problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

V2 = fem.FunctionSpace(domain, ("CG", 2))
uex = fem.Function(V2)
uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)


L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))


error_max = numpy.max(numpy.abs(uD.x.array-uh.x.array))
# Only print the error on one process
if domain.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")


import pyvista
print(pyvista.global_theme.jupyter_backend)


from dolfinx import plot
topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()


u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(Myspace)

u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    u_plotter.show()