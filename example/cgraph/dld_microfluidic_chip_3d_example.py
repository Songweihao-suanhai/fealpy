import json
import fealpy.cgraph as cgraph


WORLD_GRAPH = cgraph.WORLD_GRAPH

material = cgraph.create("IncompressibleFluid")
mesher = cgraph.create("DLDMicrofluidicChipMesh3d")
physics = cgraph.create("IncompressibleFluidPhysics")
mathematics = cgraph.create("StokesMathematics")
model = cgraph.create("StokesFEMModel")
solver = cgraph.create("CGSolver")
postprocess = cgraph.create("VPDecoupling")
to_vtk = cgraph.create("TO_VTK")

material(
    material = "custom",
    mu = 1.0,
    rho = 1.0
)
mesher(lc = 0.07,
       material = material().mp
       )
physics(mesh = mesher(),
        utype = "lagrange",
        u_p = 2,
        u_gd = 3,
        ptype = "lagrange",
        p_p = 1)        
mathematics(
    u = physics().u,
    p = physics().p,
)
model(
    equation = mathematics().equation,
    boundary = mathematics().boundary,
    is_boundary = mathematics().is_boundary,
    u = physics().u,
    p = physics().p
)
solver(A = model().bform,
       b = model().lform)
postprocess(out = solver().out, 
            u = physics().u,
            p = physics().p)
to_vtk(mesh = mesher(),
        uh = (postprocess().uh, postprocess().ph),
        path = "/home/libz/dld_3d")

# 最终连接到图输出节点上
WORLD_GRAPH.output(path = to_vtk().path)
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())
