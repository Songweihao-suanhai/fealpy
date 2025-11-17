import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

WORLD_GRAPH = cgraph.WORLD_GRAPH

model = cgraph.create("TrussTower3d")
mesher = cgraph.create("TrussTowerMesh")
materialer = cgraph.create("TrussTowerMaterial")
spacer = cgraph.create("FunctionSpace")
truss_tower = cgraph.create("TrussTower")
# solver = cgraph.create("DirectSolver")
# postprocess = cgraph.create("UDecoupling")

model(
    dov=0.015,
    div=0.010,
    doo=0.010,
    dio=0.007,
    load_total=84820.0
)
mesher(
    n_panel = 19,
    Lz = 19.0,
    Wx = 0.45,
    Wy = 0.40,
    lc = 0.1,
    ne_per_bar = 1,
    face_diag = True
)
materialer(property="Steel", type="bar", E=2.0e11, nu=0.3)
spacer(type="lagrange", mesh=mesher(), p=1)
truss_tower(
    space = spacer(),
    E = materialer().E,
    mu = materialer().mu,
    load = model().external_load,
    vertical = 76,
    other = 176
)

# 最终连接到图输出节点上
WORLD_GRAPH.output(material=materialer())
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())