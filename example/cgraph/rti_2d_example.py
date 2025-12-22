import json
import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

# bm.set_backend('pytorch')
# bm.set_default_device('cpu')

WORLD_GRAPH = cgraph.WORLD_GRAPH

material = cgraph.create("RTIMaterial")
mesher = cgraph.create("RTIMesher2d")
physics = cgraph.create("CHNSPhysics")
mathmatics = cgraph.create("CHNSMathmatics")
chnsrun = cgraph.create("CHNSFEMModel")
to_vtk = cgraph.create("TO_VTK")

box = "[0.0, 1.0, 0.0, 4.0]"
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
chnsrun(
    dt = 0.00175,
    i = 0,
    equation = mathmatics().equation,
    boundary_condition = mathmatics().boundary_condition,
    is_boundary = mathmatics().is_boundary,
    apply_bc = None,
    ns_q = 3,
    ch_q = 5,
    s = 1.0,
    x0 = mathmatics().x0
)
to_vtk(mesh = mesher(),
        uh = (chnsrun().phi0,
              chnsrun().phi1,
              chnsrun().u0,
              chnsrun().u1,
              chnsrun().p1,
              chnsrun().rho),
        path = "/home/libz/Rti_2d",
        i = None)


WORLD_GRAPH.output(path = to_vtk().path)
# WORLD_GRAPH.output(mesh = mesher().mesh)
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())

