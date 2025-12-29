import fealpy.cgraph as cgraph

WORLD_GRAPH = cgraph.WORLD_GRAPH

material = cgraph.create("ElectromagneticMaterial")
point_configs = cgraph.create("PointSource")
object_configs = cgraph.create("ObjectSource")
mesh = cgraph.create("YeeUniformMesh")
fdtd_solver = cgraph.create("PointMaxwellFDTDModel")

material(property="custom-input", eps=1.0, mu=1.0)
point_configs(
    source1_position=[2e-6,2.5e-6], # 源位置在中心
    source1_component="Ez",          # TM模式的Ez分量
    source1_waveform="sinusoid",     # 高斯脉冲
    source1_frequency=6e14,          # 频率参数
    source1_phase=0.0,               # 相位
    source1_amplitude=1.0,           # 振幅
    source1_spread=0,                # 展宽
    source1_injection="soft",        # 注入方式
    # source2_position=[3e-6,2e-6],      # 第二个源位置  
    # source2_component='Ez',
    # source2_waveform='sinusoid',
    # source2_frequency=3e14,
    # source2_phase=0.0,
    # source2_amplitude=2.0,
    # source2_spread=0,
    # source2_injection="soft",
)
object_configs(
    object1_box=[0,2.10e-6,2.5e-6,2.6e-6],  # 第一个物体区域
    object1_eps=10000,                 # 高介电常数物体
    object1_mu=1.0,
    object2_box=[2.35e-6,2.65e-6,2.5e-6,2.6e-6], # 无第二个物体
    object2_eps=10000,
    object2_mu=1.0,
    object3_box=[2.9e-6,5e-6,2.5e-6,2.6e-6],
    object3_eps=10000,
    object3_mu=1.0
)
mesh(
    domain=[0, 5e-6, 0, 5e-6],
    n=50,
    mp=material(),
    source_configs=point_configs(),
    object_configs=object_configs()
)

fdtd_solver(
    mesh=mesh(),
    dt=None,              # 自动计算时间步长
    maxstep=200,         # 1000个时间步
    save_every=1,        # 每1步保存一次
    boundary="UPML",      # UPML吸收边界
    pml_width=8,          # PML层宽度
    pml_m=5.0,             # PML参数
    output_dir="/home/libz/fdtd/"  # 输出目录
)

# 设置输出
WORLD_GRAPH.output(mesh=mesh(), field_history=fdtd_solver().field_history)
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())