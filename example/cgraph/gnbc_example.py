import json
import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm
bm.set_backend("pytorch")
WORLD_GRAPH = cgraph.WORLD_GRAPH
mesher = cgraph.create("PhaseFieldMesher2d")
physics = cgraph.create("MultiphaseFlowPhysics")
material = cgraph.create("MultiphaseFlowMaterial")
mathmatics = cgraph.create("CHNSMathmatics")

simulation = cgraph.create("GNBCSimulation")
simulation2 = cgraph.create("GNBCSimulation")
simulation3 = cgraph.create("GNBCSimulation")
GNBC = cgraph.create("GNBC")
to_vtk = cgraph.create("TO_VTK")
to_vtk2 = cgraph.create("TO_VTK")
to_vtk3 = cgraph.create("TO_VTK")


box = "[-0.5, 0.5, -0.125, 0.125]"


material(
    Re = 5.0,
    lam = 12.0,
    gamma = 0.0005,
    epsilon = 0.004,
    L_s = 0.0000025,
    V_s = 200.0 
)

mesher(
    material = material().material,
    box = box,
    nx = 256,
    ny = 64
)


physics(
    mesh = mesher().mesh,
    phitype = "lagrange",
    phi_p = 1,
    utype = "lagrange",
    u_p = 2,
    u_gd = 2,
    ptype = "lagrange",
    p_p = 0,
    p_ctype = 'D'
)
mathmatics(
    phi = physics().phi,
    u = physics().u,
    p = physics().p,
    VariableDensity = False
)


GNBC(equation = mathmatics().equation,
    u = physics().u,
    p = physics().p,
    dt = 0.000390625,
    is_wall_boundary = mathmatics().is_boundary,
    q = 5
)
simulation(
    dt = 0.000390625,
    i = 0,
    equation = mathmatics().equation,
    is_wall_boundary = mathmatics().is_boundary,
    phi = physics().phi,
    p = physics().p,
    u = physics().u,
    NS_BC = GNBC().apply_bc, 
    q = 5)

to_vtk(mesh = mesher(),
        uh = (simulation().u1, simulation().p1, simulation().phi1, simulation().mu1),
        path = "/home/edwin/output",
        i = 0)

simulation2(
    dt = 0.000390625,
    i = 1,
    equation = mathmatics().equation,
    is_wall_boundary = mathmatics().is_boundary,
    phi = physics().phi,
    p = physics().p,
    u = physics().u,
    NS_BC = GNBC().apply_bc,
    u0 = simulation().u0,
    u1 = simulation().u1,
    phi0 = simulation().phi0,
    phi1 = simulation().phi1, 
    q = 5)


to_vtk2(mesh = mesher(),
        uh = (simulation2().u1, simulation2().p1, simulation2().phi1, simulation2().mu1),
        path = "/home/edwin/output",
        i = 1)

simulation3(
    dt = 0.000390625,
    i = 1,
    equation = mathmatics().equation,
    is_wall_boundary = mathmatics().is_boundary,
    phi = physics().phi,
    p = physics().p,
    u = physics().u,
    NS_BC = GNBC().apply_bc,
    u0 = simulation2().u0,
    u1 = simulation2().u1,
    phi0 = simulation2().phi0,
    phi1 = simulation2().phi1, 
    q = 5)


to_vtk3(mesh = mesher(),
        uh = (simulation3().u1, simulation3().p1, simulation3().phi1, simulation3().mu1),
        path = "/home/edwin/output",
        i = 0)

WORLD_GRAPH.output(path = to_vtk3().path)
                

# 最终连接到图输出节点上
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())