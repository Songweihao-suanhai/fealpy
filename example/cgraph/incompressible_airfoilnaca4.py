import fealpy.cgraph as cgraph


WORLD_GRAPH = cgraph.WORLD_GRAPH

material = cgraph.create("IncompressibleFluid")
geometry = cgraph.create("NACA4Geometry2d")
mesher = cgraph.create("NACA4Mesh2d")
physics = cgraph.create("IncompressibleFluidPhysics")
mathmatics = cgraph.create("IncompressibleNSMathematics")
IncompressibleNSRun = cgraph.create("IncompressibleNSFEMModel")
to_vtk = cgraph.create("TO_VTK")

box = '[-0.5, 2.7, -0.4, 0.4]'
material(
    mu = 0.001,
    rho = 1.0
)
geometry(
    m = 0.0,
    p = 0.0,
    t = 0.12,
    c = 1.0,
    alpha = 10,
    N = 100,
    box = box,
    material = material().mp
)
mesher(
    geometry = geometry().geometry,
    h = 0.05,
    thickness = 0.005,
    ratio = 2.4,
    le_size = 0.016,
    te_size = 0.016,
    size = 0.001
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
    equation = mathmatics().equation,
    boundary = mathmatics().boundary,
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
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())

