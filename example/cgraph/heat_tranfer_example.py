import json
import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm
bm.set_backend("pytorch")
WORLD_GRAPH = cgraph.WORLD_GRAPH
generation = cgraph.create("HeatTransferParticleGeneration")
SPHQuery1 = cgraph.create("SPHQuery")
iterative1 = cgraph.create("HeatTransferParticleIterativeUpdate")
SPHQuery2 = cgraph.create("SPHQuery")
iterative2 = cgraph.create("HeatTransferParticleIterativeUpdate")
SPHQuery3 = cgraph.create("SPHQuery")
iterative3 = cgraph.create("HeatTransferParticleIterativeUpdate")
generation(
    dx=0.02,
    dy=0.02,
)
SPHQuery1(
    mesh = generation().mesh,
    kernel = "quintic",
    dx=0.02,
    space = True,
    box_size = generation().box_size,
)
iterative1(mesh = generation().mesh,
        i=0,
        box_size = generation().box_size,
        dx=0.02, 
        dt=0.00045454545454545455,
        neighbors=SPHQuery1().neighbors,
        self_node=SPHQuery1().self_node,
        dr=SPHQuery1().dr,
        dist=SPHQuery1().dist,
        w=SPHQuery1().w,
        grad_w=SPHQuery1().grad_w,
        grad_w_norm=SPHQuery1().grad_w_norm,
        output_dir="/home/edwin/output")

SPHQuery2(
    mesh = iterative1().mesh,
    kernel = "quintic",
    dx=0.02,
    space = True,
    box_size = generation().box_size,
)
iterative2(mesh = iterative1().mesh,
        i=1,
        box_size = generation().box_size,
        dx=0.02, 
        dt=0.00045454545454545455,
        neighbors=SPHQuery2().neighbors,
        self_node=SPHQuery2().self_node,
        dr=SPHQuery2().dr,
        dist=SPHQuery2().dist,
        w=SPHQuery2().w,
        grad_w=SPHQuery2().grad_w,
        grad_w_norm=SPHQuery2().grad_w_norm,
        output_dir="/home/edwin/output")

SPHQuery3(
    mesh = iterative2().mesh,
    kernel = "quintic",
    dx=0.02,
    space = True,
    box_size = generation().box_size,
)
iterative3(mesh = iterative2().mesh,
        i=2,
        box_size = generation().box_size,
        dx=0.02, 
        dt=0.00045454545454545455,
        neighbors=SPHQuery3().neighbors,
        self_node=SPHQuery3().self_node,
        dr=SPHQuery3().dr,
        dist=SPHQuery3().dist,
        w=SPHQuery3().w,
        grad_w=SPHQuery3().grad_w,
        grad_w_norm=SPHQuery3().grad_w_norm,
        output_dir="/home/edwin/output")


WORLD_GRAPH.output(velocity=iterative3().velocity, 
                   pressure=iterative3().pressure,
                   temperature=iterative3().temperature)

# 最终连接到图输出节点上
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())