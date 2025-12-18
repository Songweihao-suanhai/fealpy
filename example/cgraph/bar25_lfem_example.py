import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

WORLD_GRAPH = cgraph.WORLD_GRAPH

mesher = cgraph.create("Bar25TrussModel")
spacer = cgraph.create("FunctionSpace")
assembly = cgraph.create("BarStiffnessAssembly")
boundary = cgraph.create("BoundaryCondition")
solver = cgraph.create("DirectSolver")
postprocess = cgraph.create("UDecoupling")
coord = cgraph.create("Rbar3d")
strain_stress = cgraph.create("BarStrainStress")

mesher(A = 2000, E = 1500, nu=0.3,
    fx = 0.0, fy = 900.0, fz = 0.0
        )
spacer(type="lagrange", mesh=mesher(), p=1)
assembly(mesh=mesher())
boundary(mesh=mesher(), K=assembly(), method="direct")

solver(A = boundary().K_bc,
       b = boundary().F_bc)

postprocess(out=solver().out, node_ldof=3, type="Truss")

coord(mesh=mesher(), index=None)
strain_stress(
    mesh=mesher(),
    uh=solver().out,
    coord_transform=coord().R
)

# to_vtk(mesh = mesher(),
#         uh = (postprocess().uh, strain_stress().stress),
#         path = "C:/Users/Administrator/Desktop/truss/")

# 最终连接到图输出节点上
WORLD_GRAPH.output(mesh=mesher())
WORLD_GRAPH.output(uh=postprocess().uh, 
                   strain=strain_stress().strain, stress=strain_stress().stress
                   )
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())