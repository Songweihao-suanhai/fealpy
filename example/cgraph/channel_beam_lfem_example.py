import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

WORLD_GRAPH = cgraph.WORLD_GRAPH
mesher = cgraph.create("ChannelBeamModel")
spacer = cgraph.create("FunctionSpace")

# solver = cgraph.create("DirectSolver")
# postprocess = cgraph.create("UDecoupling")
# coord = cgraph.create("Rbeam3d")
# strain_stress = cgraph.create("ChannelStrainStress")

# 连接节点
mesher(L=1.0,
       n=10,
       E=2.1e9,
       nu=0.25,
       rho=7800.0,
       mu_y=2.44,
       mu_z=2.38,
       load_case=1,
       F_x=10.0,
       F_y=50.0,
       F_z=100.0,
       M_x=-10.0)
spacer(type="lagrange", mesh=mesher(), p=1)


# solver(A = ChannelBeam_model().K,
#        b = ChannelBeam_model().F)

# postprocess(out = solver().out, node_ldof=6, type="Timo_beam")

# coord(mesh=mesher(), vref=[0, 1, 0], index=None)
# strain_stress(
#     mesh=mesher(),
#     uh = solver().out,
#     coord_transform=coord().R,
#     y = 0.0,
#     z = 0.0
# )


# 最终连接到图输出节点上
WORLD_GRAPH.output(mesher=mesher())
# WORLD_GRAPH.output(out=solver().out,
#                    strain=strain_stress().strain, stress=strain_stress().stress
#                    )
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())