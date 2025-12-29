
import fealpy.cgraph as cgraph

WORLD_GRAPH = cgraph.WORLD_GRAPH
mesher = cgraph.create("InpMeshReader") 
eig_eq = cgraph.create("GearBox")
eigensolver = cgraph.create("SLEPcEigenSolver")
postprocess = cgraph.create("GearboxPostprocess")

mesher(input_inp_file ='/home/hk/下载/box_case3.inp') 
eig_eq(mesh=mesher())


eigensolver(
    S=eig_eq().stiffness,
    M=eig_eq().mass,
    neigen=6,
)

postprocess(
    vecs=eigensolver().vec,
    vals=eigensolver().val,
    mesh=eig_eq().mesh,
    output_file="/home/hk/Fealpy"
)

WORLD_GRAPH.output(freqs=postprocess().freqs, vecs=postprocess().eigvecs)
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())