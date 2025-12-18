import fealpy.cgraph as cgraph


WORLD_GRAPH = cgraph.WORLD_GRAPH

material = cgraph.create("IncompressibleFluid")
mesher = cgraph.create("NACA4Mesh2d")
physics = cgraph.create("IncompressibleNSPhysics")
mathmatics = cgraph.create("IncompressibleNSMathematics")
IncompressibleNSRun = cgraph.create("IncompressibleNSFEMModel")
to_vtk = cgraph.create("TO_VTK")

box = '[-0.5, 2.7, -0.4, 0.4]'
material(
    mu = 0.001,
    rho = 1.0
)
mesher(
    m = 0.0,
    p = 0.0,
    t = 0.12,
    c = 1.0,
    alpha = 0.0,
    N = 100,
    box = box,
    h = 0.05,
    material = material().material
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
    u = physics().u,
    p = physics().p,
    velocity_boundary = "[(y+0.4)*(0.4-y), 0.0]",
    pressure_boundary = 0.0,
    velocity_0 = 0.0,
    pressure_0 = 0.0,
)
IncompressibleNSRun(
    dt = 0.0001,
    i = 0,
    method_name = "IPCS",
    time_derivative = mathmatics().time_derivative,
    convection = mathmatics().convection,
    pressure = mathmatics().pressure,
    viscosity = mathmatics().viscosity,
    source = mathmatics().source,
    dirichlet_boundary = mathmatics().dirichlet_boundary,
    is_boundary = mathmatics().is_boundary,
    q = 3,
    uh0 = mathmatics().u0,
    ph0 = mathmatics().p0,
)
to_vtk(mesh = mesher(),
        uh = (IncompressibleNSRun().uh, IncompressibleNSRun().ph),
        path = "/home/libz/naca4",
        i = None)

WORLD_GRAPH.output(path = to_vtk().path)
# WORLD_GRAPH.output(dirichlet_boundary = mathmatics().dirichlet_boundary)
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())

