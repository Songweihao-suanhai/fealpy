
from fealpy import logger
import fealpy.cgraph as cgraph

logger.setLevel("DEBUG")
air_material = cgraph.create("ElectromagneticMaterial")
sub_material = cgraph.create("ElectromagneticMaterial")
mesher = cgraph.create("MicrostripPatchMesher3d")
msa = cgraph.create("TimeHarmonicMaxwellWithLumpedPort")
solver = cgraph.create("IterativeSolver")
postprocess = cgraph.create("AntennaPostprocess")

msa(
    mesh=mesher().mesh,
    p=0,
    f=1.575,
    air_material=air_material(mu=1.0, eps=1.0),
    sub_material=sub_material(mu=1.0, eps=3.38),
    r0=100.0,
    r1=120.0
)
solver(
    A=msa().operator,
    b=msa().vector,
    solver="minres",
    rtol=1e-8,
    atol=1e-12
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
