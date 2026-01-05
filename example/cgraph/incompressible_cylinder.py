import json
import fealpy.cgraph as cgraph


WORLD_GRAPH = cgraph.WORLD_GRAPH

material = cgraph.create("IncompressibleFluid")
mesher = cgraph.create("FlowPastCylinder2d")
physics = cgraph.create("IncompressibleFluidPhysics")
mathmatics = cgraph.create("IncompressibleNSMathematics")
model = cgraph.create("IncompressibleNSFEMModel")
to_vtk = cgraph.create("TO_VTK")

material(
    mu = 0.001,
    rho = 1.0
)
box = '[0.0, 2.2, 0.0, 0.41]'
mesher(
    box = box,
    center = '(0.2, 0.2)',
    radius = 0.05,
    n_circle = 100,
    h = 0.06,
    material = material().mp
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
    velocity_boundary = "[y*(0.41 - y)*1.5, 0.0]",
    pressure_boundary = "0.0",
    velocity_0 = 0.0,
    pressure_0 = 0.0,
)
model(
    dt = 0.001,
    i = 0,
    method_name = "IPCS",
    equation = mathmatics().equation,
    boundary = mathmatics().boundary,
    x0 = mathmatics().x0,
)
to_vtk(mesh = mesher(),
        uh = (model().uh, model().ph),
        path = "/home/libz/cylinder2d",
        i = None)

WORLD_GRAPH.output(path = to_vtk().path)
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())

