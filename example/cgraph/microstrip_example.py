
from fealpy import logger
import fealpy.cgraph as cgraph

logger.setLevel("DEBUG")
mesher = cgraph.create("MicrostripPatchMesher3d")
msa = cgraph.create("TimeHarmonicMaxwellWithLumpedPort")
solver = cgraph.create("IterativeSolver")
postprocess = cgraph.create("AntennaPostprocess")

msa(
    mesh=mesher().mesh,
    p=0,
    f=1.575,
    sub_region=mesher().sub,
    air_region=mesher().air,
    pec_face=mesher().pec,
    lumped_edge=mesher().lumped,
    r0=100.0,
    r1=120.0
)
solver(
    A=msa().operator,
    b=msa().vector,
    solver="minres"
)
postprocess(
    uh=solver(),
    mesh=mesher().mesh,
    p=0
)

g1 = cgraph.Graph()
g1.output(
    e=postprocess().E,
    # sub=mesher().sub,
    # space=msa().space,
    mesh=mesher().mesh,
    # A=msa().operator
)
g1.register_error_hook(lambda x: print(x.traceback))
g1.execute()
result = g1.get()

mesh = result["mesh"]
E = result["e"]
mesh.celldata['E'] = E
mesh.to_vtk("msmesh.vtu")
