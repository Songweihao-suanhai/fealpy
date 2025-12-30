import json
import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm
bm.set_backend("numpy")
WORLD_GRAPH = cgraph.WORLD_GRAPH
generation = cgraph.create("DamBreak3DParticleGeneration")
SPHQuery1 = cgraph.create("SPHQueryDam")
iterative1 = cgraph.create("DamBreak3DParticleIterativeUpdate")
SPHQuery2 = cgraph.create("SPHQueryDam")
iterative2 = cgraph.create("DamBreak3DParticleIterativeUpdate")
SPHQuery3 = cgraph.create("SPHQueryDam")
iterative3 = cgraph.create("DamBreak3DParticleIterativeUpdate")
generation(
    dx=0.02,
    dy=0.02,
    dz=0.02,
)
SPHQuery1(
    mesh = generation().mesh,
    kernel = "wendlandc2",
    space = False,
    box_size = generation().box_size,
)
iterative1(mesh = generation().mesh,
        i=0,
        dt=0.001,
        neighbors=SPHQuery1().neighbors,
        self_node=SPHQuery1().self_node,
        dr=SPHQuery1().dr,
        dist=SPHQuery1().dist,
        w=SPHQuery1().w,
        grad_w=SPHQuery1().grad_w,
        output_dir="/home/peter/")
SPHQuery2(
    mesh = iterative1().mesh,
    kernel = "wendlandc2",
    space = False,
    box_size = generation().box_size,
)
iterative2(mesh = iterative1().mesh,
        i=1,
        dt=0.001,
        neighbors=SPHQuery2().neighbors,
        self_node=SPHQuery2().self_node,
        dr=SPHQuery2().dr,
        dist=SPHQuery2().dist,
        w=SPHQuery2().w,
        grad_w=SPHQuery2().grad_w,
        output_dir="/home/peter/")

SPHQuery3(
    mesh = iterative2().mesh,
    kernel = "wendlandc2",
    space = False,
    box_size = generation().box_size,
)
iterative3(mesh = iterative2().mesh,
        i=2,
        dt=0.001,
        neighbors=SPHQuery3().neighbors,
        self_node=SPHQuery3().self_node,
        dr=SPHQuery3().dr,
        dist=SPHQuery3().dist,
        w=SPHQuery3().w,
        grad_w=SPHQuery3().grad_w,
        output_dir="/home/peter/")


WORLD_GRAPH.output(velocity=iterative3().velocity, 
                   pressure=iterative3().pressure,)

# 最终连接到图输出节点上
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())
