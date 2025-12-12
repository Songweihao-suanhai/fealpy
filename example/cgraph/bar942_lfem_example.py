import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

WORLD_GRAPH = cgraph.WORLD_GRAPH

geometry = cgraph.create("Bar942Geometry")
mesher = cgraph.create("CreateMesh")
section = cgraph.create("PredefinedSection")
load = cgraph.create("Bar942Load")
const = cgraph.create("Bar942Boundary")
spacer = cgraph.create("FunctionSpace")
materialer = cgraph.create("BarMaterial")
# bar942_model = cgraph.create("BarModel")
# solver = cgraph.create("DirectSolver")
# postprocess = cgraph.create("UDecoupling")
# coord = cgraph.create("Rbar3d")
# strain_stress = cgraph.create("BarStrainStress")

geometry(d1 = 2135,
        d2 = 5335,
        d3 = 7470,
        d4 = 9605,
        r2 = 4265,
        r3 = 6400,
        r4 = 8535,
        l3 = 43890,
        l2 = None,
        l1 = None
        )
mesher(node=geometry().node, cell=geometry().cell)
section(section_source="bar942")
load(mesh=mesher())
const(mesh=mesher())
spacer(type="lagrange", mesh=mesher(), p=1)
materialer(bar_type="bar942")
# bar942_model(
#     bar_type="bar942",
#     space_type="lagrangespace",
#     GD = model942().GD,
#     mesh = mesher942(),
#     E = materialer942().E,
#     nu = materialer942().nu,
#     external_load = model942().external_load,
#     dirichlet_dof = model942().dirichlet_dof,
#     dirichlet_bc = model942().dirichlet_bc,
#     penalty = 1e12
# )

# solver(A = bar942_model().K,
#        b = bar942_model().F)

# postprocess(out = solver().out, node_ldof=3, type="Truss")
# coord(mesh=mesher942(), index=None)

# strain_stress(
#     bar_type="bar942",
#     E = materialer942().E,
#     nu = materialer942().nu,
#     mesh = mesher942(),
#     uh = solver().out,
#     coord_transform = coord().R
# )

# 最终连接到图输出节点上
WORLD_GRAPH.output(mesh=mesher(), A=section(), 
                   E=materialer().E, nu=materialer().nu)
# WORLD_GRAPH.output(uh=postprocess().uh, 
#                    strain=strain_stress().strain, stress=strain_stress().stress
#                    )
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())