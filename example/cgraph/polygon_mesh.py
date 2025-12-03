import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

WORLD_GRAPH = cgraph.WORLD_GRAPH

mesh = cgraph.create("PolygonMesh2d")

mesh(mesh_type="triangle", vertices=[(0,0), (1,0), (1,1), (0,1)], h=0.1)
# mesh(mesh_type="quadrangle", vertices=[(0,0), (1,0), (1,1), (0,1)], h=0.1)

# 最终连接到图输出节点上
WORLD_GRAPH.output(mesh=mesh())
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())