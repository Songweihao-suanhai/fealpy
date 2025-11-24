import fealpy.cgraph as cgraph

WORLD_GRAPH = cgraph.WORLD_GRAPH

# 创建电磁场问题定义节点
pde = cgraph.create("PointSourceMaxwell")
# 创建网格节点
mesh_gen = cgraph.create("YeeUniformMesh")
# 创建FDTD求解器节点
fdtd_solver = cgraph.create("PointSourceMaxwellFDTDModel")

# 配置PDE模型
pde(
    domain=[0, 1, 0, 1],  # 2D计算域
    eps=1.0,              # 相对介电常数
    mu=1.0,               # 相对磁导率
    source_position=[0.5, 0.5],      # 源位置在中心
    source_component="Ez",           # TM模式的Ez分量
    source_waveform="sinusoid",      # 高斯脉冲
    source_amplitude=1.0,            # 源幅度
    source_spread=0,                 # 点源
    source_injection="soft",         # 软注入
    object1_box=[0.3, 0.4, 0.3, 0.4],  # 第一个物体区域
    object1_eps=4.0,                 # 高介电常数物体
    object1_mu=1.0,
    object2_box=None                 # 无第二个物体
)

# 配置网格
mesh_gen(
    domain=pde().domain,  # 使用PDE的计算域
    n=100                 # 100x100网格
)

# 配置FDTD求解器
fdtd_solver(
    eps=pde().eps,
    mu=pde().mu, 
    mesh=mesh_gen(),
    source_config=pde().source_config,
    object_configs=pde().object_configs,
    dt=None,              # 自动计算时间步长
    maxstep=1000,         # 1000个时间步
    save_every=1,        # 每1步保存一次
    boundary="UPML",      # UPML吸收边界
    pml_width=8,          # PML层宽度
    pml_m=5.0             # PML参数
)

# 设置输出
WORLD_GRAPH.output(field_history=fdtd_solver().field_history)

# 执行计算图
WORLD_GRAPH.execute()
