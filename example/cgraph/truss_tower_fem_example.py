import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

WORLD_GRAPH = cgraph.WORLD_GRAPH

mesher = cgraph.create("TrussTowerModel")
spacer = cgraph.create("FunctionSpace")
assembly = cgraph.create("BarStiffnessAssembly")
boundary = cgraph.create("BoundaryCondition")
solver = cgraph.create("DirectSolver")
postprocess = cgraph.create("UDecoupling")
coord = cgraph.create("Rbar3d")
strain_stress = cgraph.create("BarStrainStress")

mesher(
    n_panel = 19,
    Lz = 19.0,
    Wx = 0.45,
    Wy = 0.40,
    lc = 0.1,
    ne_per_bar = 1,
    face_diag = True,
    vertical_D_outer = 0.015,
    vertical_D_inner = 0.010,
    other_D_outer = 0.010,
    other_D_inner = 0.007,
    E = 2.0e11,
    nu = 0.3,
    load_case = 1,
    load_value = 1.0    
)
spacer(type="lagrange", mesh=mesher(), p=1)
assembly(mesh=mesher())
boundary(mesh=mesher(), K=assembly(), method="direct")

solver(A = boundary().K_bc,
       b = boundary().F_bc)

postprocess(out = solver().out, node_ldof=3, type="Truss")
coord(mesh=mesher())
strain_stress(
    mesh = mesher(),
    uh = solver().out,
    coord_transform = coord().R
)

# 最终连接到图输出节点上
WORLD_GRAPH.output(uh=postprocess().uh, strain=strain_stress().strain, stress=strain_stress().stress
                )
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())