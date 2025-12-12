import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

WORLD_GRAPH = cgraph.WORLD_GRAPH

geometry = cgraph.create("Bar25Geometry")
mesher = cgraph.create("CreateMesh")
section = cgraph.create("PredefinedSection")
load = cgraph.create("Bar25Load")
const = cgraph.create("Bar25Boundary")
spacer = cgraph.create("FunctionSpace")
materialer = cgraph.create("BarMaterial")
assembly = cgraph.create("BarStiffnessAssembly")
# bar25_model = cgraph.create("BarModel")
# solver = cgraph.create("DirectSolver")
# postprocess = cgraph.create("UDecoupling")
# coord = cgraph.create("Rbar3d")
# strain_stress = cgraph.create("BarStrainStress")
# to_vtk = cgraph.create("TO_VTK")

geometry()
mesher(node=geometry().node, cell=geometry().cell)
section(section_source="bar25")
load(mesh=mesher())
const(mesh=mesher())
spacer(type="lagrange", mesh=mesher(), p=1)
materialer(bar_type="bar25")
assembly(
    mesh=mesher(),
    GD=3,
    E=materialer().E,
    area=section().area
)

# bar25_model(
#     bar_type="bar25",
#     space_type="lagrangespace",
#     GD = model25().GD,
#     mesh = mesher25(),
#     E = materialer25().E,
#     nu = materialer25().nu,
#     external_load = model25().external_load,
#     dirichlet_dof = model25().dirichlet_dof,
#     dirichlet_bc = model25().dirichlet_bc
# )

# solver(A = bar25_model().K,
#        b = bar25_model().F)

# postprocess(out = solver().out, node_ldof=3, type="Truss")
# coord(mesh=mesher25(), index=None)

# strain_stress(
#     bar_type="bar25",
#     E = materialer25().E,
#     nu = materialer25().nu,
#     mesh = mesher25(),
#     uh = solver().out,
#     coord_transform = coord().R
# )

# to_vtk(mesh = mesher25(),
#         uh = (postprocess().uh, strain_stress().stress),
#         path = "C:/Users/Administrator/Desktop/truss/")

# 最终连接到图输出节点上
WORLD_GRAPH.output(A=section(), E=materialer().E, nu=materialer().nu,
                   K=assembly().K)
# WORLD_GRAPH.output(uh=postprocess().uh, 
#                    strain=strain_stress().strain, stress=strain_stress().stress
#                    )
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())