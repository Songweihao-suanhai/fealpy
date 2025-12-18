from ...nodetype import CNodeType, PortConf, DataType

__all__ = ["Bar25TrussModel", "Bar942TrussModel", "TrussTowerModel"]


class Bar25TrussModel(CNodeType):
    r"""25-bar truss mesh generator with complete physical properties.

    Generates node coordinates and cell connectivity for the classic 25-bar 
    space truss structure. 
    
    Inputs:
        A (FLOAT): Cross-sectional area of truss members [mm²].
        E (FLOAT): Young's modulus [MPa].
        nu (FLOAT): Poisson's ratio.
        fx (FLOAT): Load in x-direction at top nodes (Z=5080) [N].
        fy (FLOAT): Load in y-direction at top nodes (Z=5080) [N].
        fz (FLOAT): Load in z-direction at top nodes (Z=5080) [N].
        
    Outputs:
        mesh (MESH): Edge mesh with nodedata and celldata containing physical properties.
    """
    TITLE: str = "25杆桁架模型"
    PATH: str = "examples.CSM"
    INPUT_SLOTS = [
        PortConf("A", DataType.FLOAT, 0, desc="桁架杆件横截面积", title="截面面积", default=2000.0),
        PortConf("E", DataType.FLOAT, 0, desc="杨氏模量", title="弹性模量", default=1500.0),
        PortConf("nu", DataType.FLOAT, 0, desc="泊松比", title="泊松比", default=0.3),
        PortConf("fx", DataType.FLOAT, 0, desc="顶部节点(Z=5080)X方向载荷", title="X向载荷", default=0.0),
        PortConf("fy", DataType.FLOAT, 0, desc="顶部节点(Z=5080)Y方向载荷", title="Y向载荷", default=900.0),
        PortConf("fz", DataType.FLOAT, 0, desc="顶部节点(Z=5080)Z方向载荷", title="Z向载荷", default=0.0)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import bm
        from fealpy.mesh import EdgeMesh
        
        # Standard 25-bar truss node coordinates (unit: mm)
        node = bm.array([
            [-950, 0, 5080],      # Node 0: top
            [950, 0, 5080],       # Node 1: top
            [-950, 950, 2540],    # Node 2: middle layer
            [950, 950, 2540],     # Node 3: middle layer
            [950, -950, 2540],    # Node 4: middle layer
            [-950, -950, 2540],   # Node 5: middle layer
            [-2540, 2540, 0],     # Node 6: bottom support
            [2540, 2540, 0],      # Node 7: bottom support
            [2540, -2540, 0],     # Node 8: bottom support
            [-2540, -2540, 0]     # Node 9: bottom support
        ], dtype=bm.float64)
        
        # Cell connectivity relations
        cell = bm.array([
            [0, 1], [3, 0], [1, 2], [1, 5], [0, 4],     # Top layer bars
            [1, 3], [1, 4], [0, 2], [0, 5], [2, 5],     # Middle connection bars
            [4, 3], [2, 3], [4, 5],                      # Middle layer bars
            [2, 9], [6, 5], [8, 3], [7, 4],             # Middle to bottom
            [6, 3], [2, 7], [9, 4], [8, 5],             # Diagonal bracing bars
            [9, 5], [2, 6], [7, 3], [8, 4]              # Bottom diagonal bracing
        ], dtype=bm.int32)
        
        mesh = EdgeMesh(node, cell)
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        
        # 材料属性
        A = options.get("A")
        E = options.get("E")
        nu = options.get("nu")
        
        mesh.celldata['A'] = bm.full(NC, A, dtype=bm.float64)
        mesh.celldata['E'] = bm.full(NC, E, dtype=bm.float64)
        mesh.celldata['nu'] = bm.full(NC, nu, dtype=bm.float64)

        # 载荷: 在Z=5080高度的节点(节点0,1)上施加Y方向900N的力
        fx = options.get("fx")
        fy = options.get("fy")
        fz = options.get("fz")
        
        load = bm.zeros((NN, 3), dtype=bm.float64)
        load[0:2, 0] = fx  # 节点0,1的x方向载荷
        load[0:2, 1] = fy  # 节点0,1的y方向载荷
        load[0:2, 2] = fz  # 节点0,1的z方向载荷
        mesh.nodedata['load'] = load
        
        # 约束: 固定底部四个节点(节点6,7,8,9)的所有自由度
        # constraint格式: [node_idx, flag_x, flag_y, flag_z]
        constraint = bm.zeros((NN, 4), dtype=bm.float64)
        constraint[:, 0] = bm.arange(NN, dtype=bm.float64)
        
        # 固定节点6,7,8,9(索引)的所有自由度
        constrained_nodes = bm.array([6, 7, 8, 9], dtype=bm.int32)
        constraint[constrained_nodes, 1:4] = 1.0  # x,y,z方向全部固定
        mesh.nodedata['constraint'] = constraint
        
        return mesh


class Bar942TrussModel(CNodeType):
    r"""942-bar truss mesh generator with complete physical properties.

    Generates the classic 942-bar space truss tower structure mesh with:
    - Geometry (node coordinates and cell connectivity)
    - Material properties (A, E) stored in mesh.celldata
    - Loads stored in mesh.nodedata
    - Constraints stored in mesh.nodedata
    
    Inputs:
        d1 (FLOAT): Half-width of first layer.
        d2 (FLOAT): Width of second layer.
        d3 (FLOAT): Width of third layer.
        d4 (FLOAT): Width of fourth layer.
        r2 (FLOAT): Radius of second layer.
        r3 (FLOAT): Radius of third layer.
        r4 (FLOAT): Radius of fourth layer.
        l3 (FLOAT): Height of third segment.
        l2 (FLOAT): Height of second segment, default l3+29260.
        l1 (FLOAT): Height of first segment, default l2+21950.
        A (FLOAT): Cross-sectional area of truss members [mm²], default 4.0.
        E (FLOAT): Young's modulus [MPa], default 2.1e5.
        fx (FLOAT): Load in x-direction at top nodes [N], default 0.0.
        fy (FLOAT): Load in y-direction at top nodes [N], default 400.0.
        fz (FLOAT): Load in z-direction at top nodes [N], default -100.0.

    Outputs:
        mesh (MESH): Edge mesh with nodedata and celldata containing physical properties.
    """
    TITLE: str = "942杆桁架模型"
    PATH: str = "examples.CSM"
    INPUT_SLOTS = [
        PortConf("d1", DataType.FLOAT, 0, desc="第一层半宽", title="第一层半宽", default=2135.0),
        PortConf("d2", DataType.FLOAT, 0, desc="第二层宽度", title="第二层宽度", default=5335.0),
        PortConf("d3", DataType.FLOAT, 0, desc="第三层宽度", title="第三层宽度", default=7470.0),
        PortConf("d4", DataType.FLOAT, 0, desc="第四层宽度", title="第四层宽度", default=9605.0),
        PortConf("r2", DataType.FLOAT, 0, desc="第二层半径", title="第二层半径", default=4265.0),
        PortConf("r3", DataType.FLOAT, 0, desc="第三层半径", title="第三层半径", default=6400.0),
        PortConf("r4", DataType.FLOAT, 0, desc="第四层半径", title="第四层半径", default=8535.0),
        PortConf("l3", DataType.FLOAT, 0, desc="第三段高度", title="第三段高度", default=43890.0),
        PortConf("l2", DataType.FLOAT, 0, desc="第二段高度", title="第二段高度", default=None),
        PortConf("l1", DataType.FLOAT, 0, desc="第一段高度", title="第一段高度", default=None),
        PortConf("A", DataType.FLOAT, 0, desc="截面面积", title="截面面积", default=4.0),
        PortConf("E", DataType.FLOAT, 0, desc="弹性模量", title="弹性模量", default=2.1e5),
        PortConf("nu", DataType.FLOAT, 0, desc="泊松比", title="泊松比", default=0.3),
        PortConf("fx", DataType.FLOAT, 0, desc="顶部节点X方向载荷", title="X向载荷", default=0.0),
        PortConf("fy", DataType.FLOAT, 0, desc="顶部节点Y方向载荷", title="Y向载荷", default=400.0),
        PortConf("fz", DataType.FLOAT, 0, desc="顶部节点Z方向载荷", title="Z向载荷", default=-100.0)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import bm
        from fealpy.mesh import EdgeMesh
        from fealpy.csm.mesh.bar942 import Bar942
        
        bar = Bar942()
        node, cell = bar.build_truss_3d(
            d1=options.get("d1"),
            d2=options.get("d2"),
            d3=options.get("d3"),
            d4=options.get("d4"),
            r2=options.get("r2"),
            r3=options.get("r3"),
            r4=options.get("r4"),
            l3=options.get("l3"),
            l2=options.get("l2"),
            l1=options.get("l1")
        )
        
        mesh = EdgeMesh(node, cell)
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        
        A = options.get("A")
        E = options.get("E")
        nu = options.get("nu")
        
        mesh.celldata['A'] = bm.full(NC, A, dtype=bm.float64)
        mesh.celldata['E'] = bm.full(NC, E, dtype=bm.float64)
        mesh.celldata['nu'] = bm.full(NC, nu, dtype=bm.float64)

        fx = options.get("fx")
        fy = options.get("fy")
        fz = options.get("fz")
        
        load = bm.zeros((NN, 3), dtype=bm.float64)
        load[0:2, 0] = fx  # 节点0,1的x方向载荷
        load[0:2, 1] = fy  # 节点0,1的y方向载荷
        load[0:2, 2] = fz  # 节点0,1的z方向载荷
        mesh.nodedata['load'] = load
        
        # constraint格式: [node_idx, flag_x, flag_y, flag_z]
        constraint = bm.zeros((NN, 4), dtype=bm.float64)
        constraint[:, 0] = bm.arange(NN, dtype=bm.float64)
        
        # 固定节点232-243(索引)的所有自由度
        constrained_nodes = bm.arange(232, min(244, NN), dtype=bm.int32)
        constraint[constrained_nodes, 1:4] = 1.0  # x,y,z方向全部固定
        mesh.nodedata['constraint'] = constraint
        
        return mesh


class TrussTowerModel(CNodeType):
    r"""Truss tower mesh generator with complete physical properties.
    
    Inputs:
       n_panel (INT): Number of panels along the z-direction (≥1).
        Lz (FLOAT): Total height of the truss tower in z-direction [m].
        Wx (FLOAT): Half-width of rectangular cross-section in x-direction [m].
        Wy (FLOAT): Half-width of rectangular cross-section in y-direction [m].
        lc (FLOAT): Characteristic geometric length for mesh size control [m].
        ne_per_bar (INT): Number of elements per bar along length (≥1).
        face_diag (BOOL): Whether to add in-plane diagonal bracing on four faces.
        vertical_D_outer (FLOAT): Outer diameter of vertical bars [mm], default 15.0.
        vertical_D_inner (FLOAT): Inner diameter of vertical bars [mm], default 10.0.
        other_D_outer (FLOAT): Outer diameter of other bars [mm], default 10.0.
        other_D_inner (FLOAT): Inner diameter of other bars [mm], default 7.0.
        E (FLOAT): Young's modulus [Pa], default 2.0e11.
        nu (FLOAT): Poisson's ratio, default 0.3.
        load_case (MENU): Load case: 1-vertical load at top, 2-quarter unit load at each top node, default 1.
        load_value (FLOAT): Load magnitude [N], default 1.0.

    Outputs:
        mesh (MESH): Edge mesh with nodedata and celldata containing physical properties.
    """
    TITLE: str = "桁架塔模型"
    PATH: str = "examples.CSM"
    INPUT_SLOTS = [
        PortConf("n_panel", DataType.INT, 0, desc="Z方向面板数量(≥1)", title="面板数量", default=19),
        PortConf("Lz", DataType.FLOAT, 0, desc="桁架塔Z方向总高度", title="总高度", default=19.0),
        PortConf("Wx", DataType.FLOAT, 0, desc="桁架块X方向宽度", title="X向宽度", default=0.45),
        PortConf("Wy", DataType.FLOAT, 0, desc="桁架块Y方向高度", title="Y向高度", default=0.40),
        PortConf("lc", DataType.FLOAT, 0, desc="网格尺寸控制特征长度", title="特征长度", default=0.1),
        PortConf("ne_per_bar", DataType.INT, 0, desc="每根杆件长度方向单元数(≥1)", title="杆件单元数", default=1),
        PortConf("face_diag", DataType.BOOL, 0, desc="是否在四个面上添加面内对角加劲", title="面内对角加劲", default=True),

        PortConf("vertical_D_outer", DataType.FLOAT, 0, desc="竖向杆外径", title="竖向杆外径", default=0.015),
        PortConf("vertical_D_inner", DataType.FLOAT, 0, desc="竖向杆内径", title="竖向杆内径", default=0.010),
        PortConf("other_D_outer", DataType.FLOAT, 0, desc="其它杆外径", title="其它杆外径", default=0.010),
        PortConf("other_D_inner", DataType.FLOAT, 0, desc="其它杆内径", title="其它杆内径", default=0.007),
        PortConf("E", DataType.FLOAT, 0, desc="弹性模量", title="弹性模量", default=2.0e11),
        PortConf("nu", DataType.FLOAT, 0, desc="泊松比", title="泊松比", default=0.3),
        PortConf("load_case", DataType.MENU, 0, desc="载荷工况:1-顶部垂直载荷,2-每个顶部节点1/4单位载荷", 
                 title="载荷工况", default=1, items={1: "工况1", 2: "工况2"}),
        PortConf("load_value", DataType.FLOAT, 0, desc="载荷大小", title="载荷大小", default=1.0)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import bm
        from fealpy.mesh import EdgeMesh
        from fealpy.csm.mesh.truss_tower import TrussTower
        
        node, cell = TrussTower.build_truss_3d_zbar(
            n_panel=options.get("n_panel"),
            Lz=options.get("Lz"),
            Wx=options.get("Wx"),
            Wy=options.get("Wy"),
            lc=options.get("lc"),
            ne_per_bar=options.get("ne_per_bar"),
            face_diag=options.get("face_diag"),
            save_msh=None
        )
        
        mesh = EdgeMesh(node, cell)
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        
        # 获取圆管截面参数
        vertical_D_outer = options.get("vertical_D_outer") 
        vertical_D_inner = options.get("vertical_D_inner") 
        other_D_outer = options.get("other_D_outer")
        other_D_inner = options.get("other_D_inner")
        
        # A = π/4 × (D_outer² - D_inner²)
        A_vertical = bm.pi / 4.0 * (vertical_D_outer**2 - vertical_D_inner**2)
        A_other = bm.pi / 4.0 * (other_D_outer**2 - other_D_inner**2)

        node0 = node[cell[:, 0]]  # shape: (NC, 3)
        node1 = node[cell[:, 1]]  # shape: (NC, 3)
        
        # 计算杆件方向向量
        bar_vec = node1 - node0  # shape: (NC, 3)
        is_vertical = (bm.abs(bar_vec[:, 0]) < 1e-6) & (bm.abs(bar_vec[:, 1]) < 1e-6)
        
        A = bm.where(is_vertical, A_vertical, A_other)
        E = options.get("E")
        nu = options.get("nu")
        
        mesh.celldata['A'] = A.astype(bm.float64)
        mesh.celldata['E'] = bm.full(NC, E, dtype=bm.float64)
        mesh.celldata['nu'] = bm.full(NC, nu, dtype=bm.float64)
        
        load_case = options.get("load_case")
        load_value = options.get("load_value")
        load = bm.zeros((NN, 3), dtype=bm.float64)
        
        z_coords = node[:, 2]
        max_z = bm.max(z_coords)
        top_nodes = bm.where(bm.abs(z_coords - max_z) < 1e-6)[0]
        
        if load_case == 1:
            # 工况1: 顶部中心施加垂直载荷 (平均分配到所有顶部节点)
            load[top_nodes, 2] = -load_value / len(top_nodes)  # 负值表示向下
        elif load_case == 2:
            # 工况2: 每个顶部节点施加四分之一单位载荷
            load[top_nodes, 2] = -load_value / 4.0  # 负值表示向下
        
        mesh.nodedata['load'] = load
        
        # 约束: 固定底部所有节点
        constraint = bm.zeros((NN, 4), dtype=bm.float64)
        constraint[:, 0] = bm.arange(NN, dtype=bm.float64)
        
        # 找到底部节点
        min_z = bm.min(z_coords)
        bottom_nodes = bm.where(bm.abs(z_coords - min_z) < 1e-6)[0]
        constraint[bottom_nodes, 1:4] = 1.0  # x,y,z方向全部固定
        mesh.nodedata['constraint'] = constraint
        
        return mesh