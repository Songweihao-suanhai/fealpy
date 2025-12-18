import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

WORLD_GRAPH = cgraph.WORLD_GRAPH

mesher = cgraph.create("TimoshenkoBeamAxleModel")
spacer = cgraph.create("FunctionSpace")
coord1 = cgraph.create("Rbeam3d")
coord2 = cgraph.create("Rbeam3d")
spring = cgraph.create("SpringStiffnessAssembly")

# solver = cgraph.create("DirectSolver")
# postprocess = cgraph.create("UDecoupling")
# strain_stress = cgraph.create("TimoAxleStrainStress")

# 连接节点
mesher(beam_para=[[120, 141, 2], [150, 28, 2], [184, 177, 4], 
                        [160, 268, 2], [184.2, 478, 2], [160, 484, 2],
                        [184, 177, 4], [150, 28, 2], [120, 141, 2]],
      axle_para=[[1.976e6, 100, 10]],
      E=2.07e11,
      nu=0.276,
      shear_factor=10/9,
      F_vertical=88200.0,
      F_axial=3140.0,
      M_torque=14000e3)
spacer(type="lagrange", mesh=mesher(), p=1)
R1 = coord1(mesh=mesher(), vref=[0, 1, 0])
R2 = coord2(mesh=mesher(), vref=[0, 1, 0])

spring(mesh=mesher(),
       R=R2,
       rotation_factor=1e3)


# solver(A = timoaxle_model().K,
#        b = timoaxle_model().F)

# postprocess(out = solver().out, node_ldof=6, type="Timo_beam")

# strain_stress(
#     mesh=mesher(), 
#     uh = solver().out,
#     y = 0.0,
#     z = 0.0,
#     axial_position = None,
#     R1 = R1,
#     R2 = R2,
#     beam_indices = indices().beam_indices,
#     axle_indices = indices().axle_indices
# )


# 最终连接到图输出节点上
WORLD_GRAPH.output(mesh=mesher())
# WORLD_GRAPH.output(out=solver().out,strain=strain_stress().strain, stress=strain_stress().stress)
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())