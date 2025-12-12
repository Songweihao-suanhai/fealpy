import json
import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

# bm.set_backend('pytorch')
# bm.set_default_device('cpu')

WORLD_GRAPH = cgraph.WORLD_GRAPH

material = cgraph.create("RTIMaterial")
mesher = cgraph.create("RTIMesher2d")
physics = cgraph.create("CHNSPhysics")
mathmatics = cgraph.create("RTIMathmatics")
bdf2 = cgraph.create("IncompressibleNSBDF2")
chfem = cgraph.create("CahnHilliardFEMSimulation")
chnsrun = cgraph.create("CHNSFEMRun")
to_vtk = cgraph.create("TO_VTK")

box = "[0, 1.0, 0.0, 4.0]"
material(
    rho_up = 3.0,
    rho_down = 1.0,
    Re = 3000.0,
    Fr = 1.0,
    epsilon = 0.01
)
mesher(
    material = material().material,
    box = box,
    nx = 64,
    ny = 256
)
physics(
    mesh = mesher().mesh,
    phitype = "lagrange",
    phi_p = 1,
    utype = "lagrange",
    u_p = 2,
    u_gd = 2,
    ptype = "lagrange",
    p_p = 1
)
mathmatics(
    phi = physics().phi,
    u = physics().u,
    p = physics().p
)
bdf2(
    u = physics().u,
    p = physics().p,
    dirichlet_boundary = None,
    is_boundary = mathmatics().is_boundary,
    q = 3
)
chfem(
    phi = physics().phi,
    q = 5,
    s = 1.0
)
chnsrun(
    dt = 0.00175,
    i = 0,
    mobility = mathmatics().mobility,
    interface = mathmatics().interface,
    free_energy = mathmatics().free_energy,
    time_derivative = mathmatics().time_derivative,
    convection = mathmatics().convection,
    pressure = mathmatics().pressure,
    viscosity = mathmatics().viscosity,
    source = mathmatics().source,
    phi0 = mathmatics().phi0,
    phi1 = mathmatics().phi1,
    u0 = mathmatics().u0,
    u1 = mathmatics().u1,
    p0 = mathmatics().p0,
    ns_update = bdf2().update,
    ch_update = chfem().update,
    is_boundary = mathmatics().is_boundary
)
to_vtk(mesh = mesher(),
        uh = (chnsrun().u, chnsrun().p, chnsrun().phi, chnsrun().rho),
        path = "/home/libz/Rti_2d",
        i = None)


WORLD_GRAPH.output(path = to_vtk().path)
# WORLD_GRAPH.output(mesh = mesher().mesh)
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())

