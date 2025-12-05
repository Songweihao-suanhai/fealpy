import json
import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

# bm.set_backend('pytorch')
# bm.set_default_device('cpu')

WORLD_GRAPH = cgraph.WORLD_GRAPH

pde = cgraph.create("RayleighTaylor")
eq = cgraph.create("CHNSEquation")
mesher = cgraph.create("Box2d")
phispacer = cgraph.create("FunctionSpace")
uspacer = cgraph.create("TensorFunctionSpace")
pspacer = cgraph.create("FunctionSpace")
bdf2 = cgraph.create("IncompressibleNSBDF2")
chfem = cgraph.create("CahnHilliardFEMSimulation")
chnsrun = cgraph.create("CHNSFEMRun")
to_vtk = cgraph.create("TO_VTK")

eq(
    rho = pde().rho,
    Re = pde().Re,
    Fr = pde().Fr,
    epsilon = pde().epsilon,
    Pe = pde().Pe,
    body_force = pde().body_force
)
mesher(
    mesh_type="triangle",
    domain=pde().domain,
    nx=64,
    ny=256
)
phispacer(
    mesh=mesher().mesh,
    p = 2
)
uspacer(
    mesh=mesher().mesh,
    p = 2,
    gd = 2
)
pspacer(
    mesh=mesher().mesh,
    p = 1
)
bdf2(
    uspace = uspacer(), 
    pspace = pspacer(),
    velocity_dirichlet = pde().velocity_dirichlet,
    pressure_dirichlet = pde().pressure_dirichlet,
    is_velocity_boundary = pde().is_velocity_boundary,
    is_pressure_boundary = pde().is_pressure_boundary,
    q = 3
)
chfem(
    phispace = phispacer(),
    q = 5,
    s = 1.0
)
chnsrun(
    dt = 0.00175,
    i = 0,
    mobility = eq().mobility,
    interface = eq().interface,
    free_energy = eq().free_energy,
    time_derivative = eq().time_derivative,
    convection = eq().convection,
    pressure = eq().pressure,
    viscosity = eq().viscosity,
    source = eq().source,
    ns_update = bdf2().update,
    ch_update = chfem().update,
    phispace = phispacer(),
    uspace = uspacer(),
    pspace = pspacer(),
    mesh = mesher(),
    init_interface = pde().init_interface,
    is_velocity_boundary = pde().is_velocity_boundary
)
to_vtk(mesh = mesher(),
        uh = (chnsrun().u, chnsrun().p, chnsrun().phi, chnsrun().rho),
        path = "/home/libz/Rti_2d",
        i = None)


WORLD_GRAPH.output(path = to_vtk().path)
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())

