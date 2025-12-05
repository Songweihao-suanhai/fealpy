import json
import fealpy.cgraph as cgraph


WORLD_GRAPH = cgraph.WORLD_GRAPH

pde = cgraph.create("FlowPastFoil")
mesher = cgraph.create("NACA4Mesh2d")
uspacer = cgraph.create("TensorFunctionSpace")
pspacer = cgraph.create("FunctionSpace")
simulation = cgraph.create("IncompressibleNSIPCS")
IncompressibleNSRun = cgraph.create("IncompressibleNSIPCSRun")
to_vtk = cgraph.create("TO_VTK")

pde(
    mu = 0.001,
    rho = 1.0,
    inflow = 2.0,
    box = [-0.5, 2.7, -0.4, 0.4]
)
mesher(
    m = 0.0,
    p = 0.0,
    t = 0.12,
    c = 1.0,
    alpha = 5.0,
    N = 100,
    box = pde().domain,
    h = 0.05
)
uspacer(mesh = mesher().mesh, p=2, gd = 2)
pspacer(mesh = mesher().mesh, p=1)
simulation(
   
    uspace = uspacer(),
    pspace = pspacer(),
    velocity_dirichlet = pde().velocity_dirichlet,
    pressure_dirichlet = pde().pressure_dirichlet,
    is_velocity_boundary = pde().is_velocity_boundary,
    is_pressure_boundary = pde().is_pressure_boundary,
    q = 3
) 
IncompressibleNSRun(
    dt = 0.01,
    i = 0,
    time_derivative = pde().rho,
    convection = pde().rho,
    pressure = 1.0,
    viscosity = pde().mu,
    source = pde().source,
    uspace = uspacer(), 
    pspace = pspacer(), 
    velocity_0 = pde().velocity_0,
    pressure_0 = pde().pressure_0,
    uh0 = None,
    ph0 = None,
    predict_velocity = simulation().predict_velocity,
    correct_pressure = simulation().correct_pressure,
    correct_velocity = simulation().correct_velocity,
)
to_vtk(mesh = mesher(),
        uh = (IncompressibleNSRun().uh, IncompressibleNSRun().ph),
        path = "/home/libz/naca4",
        i = None)

WORLD_GRAPH.output(path = to_vtk().path)
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())

