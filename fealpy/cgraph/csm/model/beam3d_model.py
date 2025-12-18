from ...nodetype import CNodeType, PortConf, DataType

__all__ = ["ChannelBeamModel", "TimoshenkoBeamAxleModel"]


class ChannelBeamModel(CNodeType):
    r"""3D Channel Beam Geometry Model with Complete Physical Properties.
    
     Generates mesh for channel beam structure with:
    - Beam elements along the length
    - Complete cross-section properties (A, I_yy, I_zz, J, W_t)
    - Shear correction factors (mu_y, mu_z)
    - Material properties (E, nu, rho)
    - Loads and constraints stored in mesh.nodedata
    
    Inputs:
        L (FLOAT): Beam length [m]. Default 1.0.
        n_elements (INT): Number of elements along the beam. Default 10.
        E (FLOAT): Young's modulus [Pa]. Default 210e9.
        nu (FLOAT): Poisson's ratio. Default 0.25.
        rho (FLOAT): Mass density [kg/m³]. Default 7800.
        mu_y (FLOAT): Ratio of max to avg shear stress for y-direction. Default 2.44.
        mu_z (FLOAT): Ratio of max to avg shear stress for z-direction. Default 2.38.
        load_case (INT): Load case (1: tip forces, 2: gravity). Default 1.
        F_X (FLOAT): Axial force at tip [N]. Default 10.0 (load_case=1).
        F_Y (FLOAT): Transverse force in y-direction at tip [N]. Default 50.0 (load_case=1).
        F_Z (FLOAT): Transverse force in z-direction at tip [N]. Default 100.0 (load_case=1).
        M_X (FLOAT): Torque at tip [N·m]. Default -10.0 (load_case=1).
    
    Outputs:
        mesh (MESH): Edge mesh with nodedata and celldata containing physical properties.
    """
    TITLE: str = "槽形梁模型"
    PATH: str = "examples.CSM"
    INPUT_SLOTS = [
        PortConf("L", DataType.FLOAT, 0, desc="梁长度", title="梁长", default=1.0),
        PortConf("n", DataType.INT, 0, desc="沿梁长方向的单元数量", title="单元剖分数", default=10),
        PortConf("E", DataType.FLOAT, 0, desc="弹性模量", title="弹性模量", default=2.1e9),
        PortConf("nu", DataType.FLOAT, 0, desc="泊松比", title="泊松比", default=0.25),
        PortConf("rho", DataType.FLOAT, 0, desc="质量密度", title="密度", default=7800.0),
        PortConf("mu_y", DataType.FLOAT, 0, desc="y方向剪切应力的最大值与平均值比例因子", title="y向剪切因子", default=2.44),
        PortConf("mu_z", DataType.FLOAT, 0, desc="z方向剪切应力的最大值与平均值比例因子", title="z向剪切因子", default=2.38),
        PortConf("load_case", DataType.MENU, 0, desc="载荷工况选择(1:端部集中力, 2:重力)", title="载荷工况", default=1, items=[1, 2]),
        PortConf("F_x", DataType.FLOAT, 0, desc="端部轴向力(工况1)", title="轴向力", default=10.0),
        PortConf("F_y", DataType.FLOAT, 0, desc="端部Y向横向力(工况1)", title="Y向力", default=50.0),
        PortConf("F_z", DataType.FLOAT, 0, desc="端部Z向横向力(工况1)", title="Z向力", default=100.0),
        PortConf("M_x", DataType.FLOAT, 0, desc="端部扭矩(工况1)", title="扭矩", default=-10.0)  
    ]
    
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import bm
        from fealpy.mesh import EdgeMesh
        
        L = options.get("L")
        n = options.get("n")
        E = options.get("E")
        nu = options.get("nu")
        rho = options.get("rho")
        mu_y = options.get("mu_y")
        mu_z = options.get("mu_z")
        load_case = options.get("load_case")
        
        # 剪切模量
        G = E * 0.5 / (1 + nu)
        
        # 槽形梁截面几何参数(固定值)
        A = 4.90e-4          # 横截面积 [m²]
        I_yy = 2.77e-8       # 弱轴惯性矩 [m⁴]
        I_zz = 1.69e-7       # 强轴惯性矩 [m⁴]
        J = 5.18e-9          # 扭转常数 [m⁴]
        W_t = 8.64e-7        # 抗扭截面模量 [m³]
        e_z = 0.0148         # 剪切中心偏移 [m]
        
        # 剪切面积(考虑剪切修正因子)
        A_y = A / mu_y
        A_z = A / mu_z

        nodes = bm.linspace(0, L, n + 1)
        node = bm.zeros((n + 1, 3), dtype=bm.float64)
        node[:, 0] = nodes  # x-coordinates along beam length

        cell = bm.zeros((n, 2), dtype=bm.int32)
        cell[:, 0] = bm.arange(n)
        cell[:, 1] = bm.arange(1, n + 1)

        mesh = EdgeMesh(node, cell)
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()

        # === 设置单元属性 ===
        mesh.celldata['type'] = bm.zeros(NC, dtype=bm.int32)  # 0 = beam
        mesh.celldata['E'] = bm.full(NC, E, dtype=bm.float64)
        mesh.celldata['G'] = bm.full(NC, G, dtype=bm.float64)
        mesh.celldata['rho'] = bm.full(NC, rho, dtype=bm.float64)
        
        # 截面几何属性
        mesh.celldata['Ax'] = bm.full(NC, A, dtype=bm.float64)
        mesh.celldata['Ay'] = bm.full(NC, A_y, dtype=bm.float64)
        mesh.celldata['Az'] = bm.full(NC, A_z, dtype=bm.float64)
        mesh.celldata['J'] = bm.full(NC, J, dtype=bm.float64)
        mesh.celldata['Iy'] = bm.full(NC, I_yy, dtype=bm.float64)
        mesh.celldata['Iz'] = bm.full(NC, I_zz, dtype=bm.float64)
        mesh.celldata['Wt'] = bm.full(NC, W_t, dtype=bm.float64)
        mesh.celldata['e_z'] = bm.full(NC, e_z, dtype=bm.float64)
        mesh.celldata['mu_y'] = bm.full(NC, mu_y, dtype=bm.float64)
        mesh.celldata['mu_z'] = bm.full(NC, mu_z, dtype=bm.float64)

        load = bm.zeros((NN, 6), dtype=bm.float64)  # [F_x, F_y, F_z, M_x, M_y, M_z]
        if load_case == 1:
            # 工况1: 端部集中力和扭矩
            F_x = options.get("F_x")
            F_y = options.get("F_y")
            F_z = options.get("F_z")
            M_x = options.get("M_x")

            load[-1, 0] = F_x  # 轴向力
            load[-1, 1] = F_y  # Y向横向力
            load[-1, 2] = F_z  # Z向横向力
            load[-1, 3] = M_x  # 扭矩
        elif load_case == 2:
            # 工况2: 重力载荷(负Z方向)
            g = 9.81  # 重力加速度 [m/s²]
            element_length = L / n
            element_mass = rho * A * element_length
            element_weight = element_mass * g
            
            # 将单元重量平均分配到两个节点
            node_weight = element_weight / 2.0
            
            # 内部节点受到相邻两个单元的重量
            load[1:-1, 2] = -2.0 * node_weight
            # 两端节点只受一个单元的重量
            load[0, 2] = -node_weight
            load[-1, 2] = -node_weight
        mesh.nodedata['load'] = load
        
        # === 设置约束 ===
        # 格式: [node_id, u_x, u_y, u_z, theta_x, theta_y, theta_z]
        constraint = bm.zeros((NN, 7), dtype=bm.float64)
        constraint[:, 0] = bm.arange(NN, dtype=bm.float64)
        constraint[0, 1:7] = 1.0
        mesh.nodedata['constraint'] = constraint
        
        # === 存储应力计算点位置 ===
        stress_points = bm.array([
            [-0.025, -0.0164],  # (y1, z1)
            [0.025, -0.0164],   # (y2, z2)
            [0.025, 0.0086],    # (y3, z3)
            [-0.025, 0.0086]    # (y4, z4)
        ], dtype=bm.float64)
        mesh.meshdata['stress_points'] = stress_points
        return mesh


class TimoshenkoBeamAxleModel(CNodeType):
    r"""3D Timoshenko Beam-Axle Geometry Model with Complete Physical Properties.
    
    Generates mesh for train axle structure with:
    - Beam elements: Based on beam_para [Diameter, Length, Count]
    - Spring elements: Based on axle_para [Stiffness, Offset_Z, Count]
    - Material properties (D, E, G, k_spring) stored in mesh.celldata
    - Loads stored in mesh.nodedata
    - Constraints stored in mesh.nodedata
    
    Inputs:
        beam_para (TENSOR): Beam section parameters, each row represents [Diameter, Length, Count].
        axle_para (TENSOR): Axle spring parameters, [Stiffness, Offset_Z, Count].
        E (FLOAT): Young's modulus [Pa]. Default 2.07e11.
        nu (FLOAT): Poisson's ratio. Default 0.276.
        F_vertical (FLOAT): Vertical load at beam segments 1 and 9 [N]. Default 88200.
        F_axial (FLOAT): Axial force at beam segment 5 [N]. Default 3140.
        M_torque (FLOAT): Torque at beam segment 5 [N·mm]. Default 14000e3.
        shear_factor (FLOAT): Shear correction factor. Default 10/9 (for circular).

    Outputs:
        mesh (MESH): Edge mesh with nodedata and celldata containing physical properties.
    """
    TITLE: str = "列车车轴模型"
    PATH: str = "examples.CSM"
    INPUT_SLOTS = [
        PortConf("beam_para", DataType.TENSOR, 0, 
                desc="梁结构参数数组，每行为 [直径, 长度, 数量]", 
                title="梁段参数",
                default=[[120, 141, 2], [150, 28, 2], [184, 177, 4], 
                        [160, 268, 2], [184.2, 478, 2], [160, 484, 2],
                        [184, 177, 4], [150, 28, 2], [120, 141, 2]]),
        PortConf("axle_para", DataType.TENSOR, 0, 
                desc="轴段弹簧参数 [刚度, Z向偏移, 数量]", 
                title="轴段参数",
                default=[[1.976e6, 100, 10]]),
        PortConf("E", DataType.FLOAT, 0, 
                desc="弹性模量", 
                title="弹性模量", 
                default=2.07e11),
        PortConf("nu", DataType.FLOAT, 0, 
                desc="泊松比", 
                title="泊松比", 
                default=0.276),
        PortConf("shear_factor", DataType.FLOAT, 0, 
                desc="剪切修正因子(圆形截面10/9, 矩形6/5)", 
                title="剪切修正因子", 
                default=10/9),
        PortConf("F_vertical", DataType.FLOAT, 0, 
                desc="轴段1和9受到的竖直向下车体重量", 
                title="竖直载荷", 
                default=88200.0),
        PortConf("F_axial", DataType.FLOAT, 0, 
                desc="轴段5受到的轴向力(向右)", 
                title="轴向力", 
                default=3140.0),
        PortConf("M_torque", DataType.FLOAT, 0, 
                desc="轴段5受到的扭矩", 
                title="扭矩", 
                default=14000e3)  
    ]
    
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import bm
        from fealpy.mesh import EdgeMesh
        
        beam_para = bm.array(options.get("beam_para"), dtype=bm.float64)
        axle_para = bm.array(options.get("axle_para"), dtype=bm.float64)
        
        E = options.get("E")
        nu = options.get("nu")
        G = E * 0.5 / (1 + nu)  # 剪切模量
        shear_factor = options.get("shear_factor")
        
        k_spring = axle_para[0, 0]  # 弹簧刚度
        spring_z = axle_para[0, 1]  # Z向偏移
        n_springs = int(axle_para[0, 2])  # 弹簧数量
        
        # === 生成梁节点坐标 ===
        cumulative_length = bm.concatenate([bm.array([0.0]), bm.cumsum(beam_para[:, 1])])
        
        node_list = []
        for i in range(len(beam_para)):
            seg = int(beam_para[i, 2])
            length = beam_para[i, 1]
            start_len = cumulative_length[i]
            
            if i == 0:
                x_coords = bm.linspace(start_len, start_len + length, seg + 1)
            else:
                # 其他段不包含起点（已被前一段的终点覆盖）
                x_coords = bm.linspace(start_len, start_len + length, seg + 1)[1:]
            node_list.append(x_coords)
            
            x_coords = bm.concatenate(node_list)
            n_beam_nodes = len(x_coords)
            base_nodes = bm.stack([x_coords, 
                    bm.zeros(n_beam_nodes, dtype=bm.float64),
                    bm.zeros(n_beam_nodes, dtype=bm.float64)], axis=1)
        
        # === 弹簧支座节点 ===   
        spring_node_indices = bm.concatenate([bm.arange(4, 9), bm.arange(14, 19)])
        spring_nodes = base_nodes[spring_node_indices].copy()
        spring_nodes[:, 2] -= spring_z  # Z 方向偏移
        
        node = bm.concatenate([base_nodes, spring_nodes], axis=0)
        NN = len(node)
        
        # 梁单元连接关系
        n_beam_cells = n_beam_nodes - 1
        beam_cells = bm.stack([bm.arange(n_beam_cells), 
                            bm.arange(1, n_beam_cells + 1)], axis=1)
        
        # 梁单元直径
        beam_diameters = bm.repeat(beam_para[:, 0], beam_para[:, 2].astype(bm.int32))
        
        spring_start_idx = n_beam_nodes
        spring_cells = bm.stack([spring_node_indices,
                                bm.arange(spring_start_idx, spring_start_idx + len(spring_node_indices))],
                                axis=1)
        
        # 合并所有单元
        cell = bm.concatenate([beam_cells, spring_cells], axis=0)
        NC = len(cell)
        
        # 单元类型: 0=beam, 1=spring
        cell_types = bm.concatenate([
            bm.zeros(n_beam_cells, dtype=bm.int32),  # beam
            bm.ones(len(spring_node_indices), dtype=bm.int32)  # spring
        ])
        
        cell_diameters = bm.concatenate([
            beam_diameters,
            bm.zeros(len(spring_node_indices), dtype=bm.float64)
        ])
        
        cell_E = bm.concatenate([
            bm.full(n_beam_cells, E, dtype=bm.float64),
            bm.zeros(len(spring_node_indices), dtype=bm.float64)
        ])
        
        cell_G = bm.concatenate([
            bm.full(n_beam_cells, G, dtype=bm.float64),
            bm.zeros(len(spring_node_indices), dtype=bm.float64)
        ])
        
        cell_k_spring = bm.concatenate([
            bm.zeros(n_beam_cells, dtype=bm.float64),
            bm.full(len(spring_node_indices), k_spring, dtype=bm.float64)
        ])
        
        mesh = EdgeMesh(node, cell)
        
        mesh.celldata['type'] = cell_types
        mesh.celldata['D'] = cell_diameters
        mesh.celldata['E'] = cell_E
        mesh.celldata['G'] = cell_G
        mesh.celldata['k_spring'] = cell_k_spring
        mesh.celldata['shear_factor'] = bm.full(NC, shear_factor, dtype=bm.float64)
        
        # 计算梁截面属性
        is_beam = cell_types == 0
        D = cell_diameters
        
        Ax = bm.where(is_beam, bm.pi * D**2 / 4, 0.0)
        Ay = bm.where(is_beam, Ax / shear_factor, 0.0)
        Az = bm.where(is_beam, Ax / shear_factor, 0.0)
        Iy = bm.where(is_beam, bm.pi * D**4 / 64, 0.0)
        Iz = Iy
        J = Iy + Iz
        
        mesh.celldata['Ax'] = Ax
        mesh.celldata['Ay'] = Ay
        mesh.celldata['Az'] = Az
        mesh.celldata['J'] = J
        mesh.celldata['Iy'] = Iy
        mesh.celldata['Iz'] = Iz
        
        # === 设置载荷 ===
        F_vertical = options.get("F_vertical")
        F_axial = options.get("F_axial")
        M_torque = options.get("M_torque")
        
        load = bm.zeros((NN, 6), dtype=bm.float64)
        load[1, 2] = -F_vertical
        load[21, 2] = -F_vertical
        load[11, 0] = F_axial
        load[11, 3] = M_torque
        mesh.nodedata['load'] = load
        
        # === 设置约束 ===
        constraint = bm.zeros((NN, 7), dtype=bm.float64)
        constraint[:, 0] = bm.arange(NN, dtype=bm.float64)
        constrained_nodes = bm.arange(spring_start_idx, NN, dtype=bm.int32)
        constraint[constrained_nodes, 1:7] = 1.0
        mesh.nodedata['constraint'] = constraint

        return mesh