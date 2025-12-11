import json
import fealpy.cgraph as cgraph


WORLD_GRAPH = cgraph.WORLD_GRAPH

mesher = cgraph.create("NACA4Mesh2d")
physics = cgraph.create("IncompressibleNSPhysics")
mathmatics = cgraph.create("IncompressibleNSMathematics")
simulation = cgraph.create("IncompressibleNSIPCS")
IncompressibleNSRun = cgraph.create("IncompressibleNSIPCSRun")
to_vtk = cgraph.create("TO_VTK")

box = "[-0.5, 2.7, -0.4, 0.4]"
mesher(
    m = 0.06,
    p = 0.4,
    t = 0.18,
    c = 1.0,
    alpha = 5.0,
    N = 100,
    box = box,
    h = 0.05
)
physics(
    mesh = mesher().mesh,
    utype = "lagrange",
    u_p = 2,
    u_gd = 2,
    ptype = "lagrange",
    p_p = 1,
)
mathmatics(
    mesh = mesher().mesh,
    u = physics().u,
    p = physics().p,
    velocity_boundary = "[1.0, 0.0]",
    pressure_boundary = 0.0,
    velocity_0 = 0.0,
    pressure_0 = 0.0,
)
simulation(
    u = physics().u,
    p = physics().p,
    dirichlet_boundary = mathmatics().dirichlet_boundary,
    is_boundary = mathmatics().is_boundary,
    q = 3
) 
IncompressibleNSRun(
    dt = 0.0001,
    i = 0,
    time_derivative = mathmatics().time_derivative,
    convection = mathmatics().convection,
    pressure = mathmatics().pressure,
    viscosity = mathmatics().viscosity,
    source = mathmatics().source,
    uh0 = mathmatics().u0,
    ph0 = mathmatics().p0,
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

