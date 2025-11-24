import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

WORLD_GRAPH = cgraph.WORLD_GRAPH
model = cgraph.create("ChannelBeam3d")
spacer = cgraph.create("FunctionSpace")
materialer = cgraph.create("ChannelBeamMaterial")
ChannelBeam_model = cgraph.create("ChannelBeam")
solver = cgraph.create("DirectSolver")
postprocess = cgraph.create("UDecoupling")
strain_stress = cgraph.create("ChannelStrainStress")

# 连接节点
spacer(type="lagrange", mesh=model().mesh, p=1)
materialer(
    property="Steel",
    beam_type="Timoshenko_beam",
    mu_y=model().mu_y,
    mu_z=model().mu_z,
    beam_E=2.1e11, beam_nu=0.3, beam_density=7800)

ChannelBeam_model(
    mu_y = model().mu_y,
    mu_z = model().mu_z,
    GD = model().GD,
    space = spacer(),
    beam_E = materialer().E,
    beam_nu = materialer().nu,
    beam_density = materialer().rho,
    load_case = 1,
    gravity = 9.81,
    dirichlet_dof = model().dirichlet_dof
)

solver(A = ChannelBeam_model().K,
       b = ChannelBeam_model().F)

# postprocess(out = solver().out, node_ldof=6, type="Timo_beam")

strain_stress(
    mu_y = model().mu_y,
    mu_z = model().mu_z,
    beam_E = materialer().E,
    beam_nu = materialer().nu,
    mesh=model().mesh,
    uh = solver().out,
    y = 0.0,
    z = 0.0
)


# 最终连接到图输出节点上
WORLD_GRAPH.output(out=solver().out, strain=strain_stress().strain, stress=strain_stress().stress)
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())