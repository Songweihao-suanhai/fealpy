import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

WORLD_GRAPH = cgraph.WORLD_GRAPH

mesher = cgraph.create("Bar942TrussModel")
spacer = cgraph.create("FunctionSpace")
assembly = cgraph.create("AssembleBarStiffness")
boundary = cgraph.create("PenaltyMethodBC")
solver = cgraph.create("DirectSolver")
postprocess = cgraph.create("UDecoupling")
coord = cgraph.create("Rbar3d")
strain_stress = cgraph.create("BarStrainStress")

mesher(d1 = 2135, d2 = 5335, d3 = 7470, d4 = 9605,
        r2 = 4265, r3 = 6400, r4 = 8535,
        l3 = 43890, l2 = None, l1 = None,
        A = 4, E = 2.1e5, nu=0.3,
        fx = 0.0, fy = 400.0, fz = -100.0
        )
spacer(type="lagrange", mesh=mesher(), p=1)
assembly(mesh=mesher())
boundary(mesh=mesher(), K=assembly(), penalty=1e12)

solver(A = boundary().K_bc,
       b = boundary().F_bc)

postprocess(out=solver().out, node_ldof=3, type="Truss")

coord(mesh=mesher(), index=None)
strain_stress(
    mesh=mesher(),
    uh=solver().out,
    coord_transform=coord().R
)

# 最终连接到图输出节点上
WORLD_GRAPH.output(mesh=mesher())
WORLD_GRAPH.output(uh=postprocess().uh, 
                   strain=strain_stress().strain, stress=strain_stress().stress
                   )
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())