from ...nodetype import CNodeType, PortConf, DataType

__all__ = ["Bar25TrussModel", "Bar942TrussModel", "TrussTowerGeometry"]


class Bar25TrussModel(CNodeType):
    r"""25-bar truss mesh generator with complete physical properties.

    Generates node coordinates and cell connectivity for the classic 25-bar 
    space truss structure. 
    
    Inputs:
        A (float): Cross-sectional area of truss members [mm²], default 1.0.
        E (float): Young's modulus [MPa], default 1500.0.
        nu (float): Poisson's ratio, default 0.3.
        fx (float): Load in x-direction at top nodes (Z=5080) [N], default 0.0.
        fy (float): Load in y-direction at top nodes (Z=5080) [N], default 900.0.
        fz (float): Load in z-direction at top nodes (Z=5080) [N], default 0.0.
        
    Outputs:
        mesh (EdgeMesh): Edge mesh with nodedata and celldata containing physical properties.
    """
    TITLE: str = "25杆桁架几何建模与网格生成"
    PATH: str = "examples.CSM"
    INPUT_SLOTS = [
        PortConf("A", DataType.FLOAT, 0, desc="Cross-sectional area of truss members [mm²]", title="截面面积", default=2000.0),
        PortConf("E", DataType.FLOAT, 0, desc="Young's modulus [MPa]", title="弹性模量", default=1500.0),
        PortConf("nu", DataType.FLOAT, 0, desc="Poisson's ratio", title="泊松比", default=0.3),
        PortConf("fx", DataType.FLOAT, 0, desc="Load in x-direction at top nodes (Z=5080) [N]", title="X向载荷", default=0.0),
        PortConf("fy", DataType.FLOAT, 0, desc="Load in y-direction at top nodes (Z=5080) [N]", title="Y向载荷", default=900.0),
        PortConf("fz", DataType.FLOAT, 0, desc="Load in z-direction at top nodes (Z=5080) [N]", title="Z向载荷", default=0.0)
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
        d1 (float): Half-width of first layer (square top) [mm].
        d2 (float): Width of second layer (octagonal section) [mm].
        d3 (float): Width of third layer (dodecagonal section) [mm].
        d4 (float): Width of fourth layer (bottom support) [mm].
        r2 (float): Radius of second layer (octagonal section) [mm].
        r3 (float): Radius of third layer (dodecagonal section) [mm].
        r4 (float): Radius of fourth layer (bottom support) [mm].
        l3 (float): Height of third segment (total dodecagonal height) [mm].
        l2 (float): Height of second segment (octagonal top height) [mm], default l3+29260.
        l1 (float): Height of first segment (square top height) [mm], default l2+21950.
        A (float): Cross-sectional area of truss members [mm²], default 4.0.
        E (float): Young's modulus [MPa], default 2.1e5.
        fx (float): Load in x-direction at top nodes [N], default 0.0.
        fy (float): Load in y-direction at top nodes [N], default 400.0.
        fz (float): Load in z-direction at top nodes [N], default -100.0.
        
    Outputs:
        mesh (EdgeMesh): Edge mesh with nodedata and celldata containing physical properties.
    """
    TITLE: str = "942杆桁架几何建模与网格生成"
    PATH: str = "examples.CSM"
    INPUT_SLOTS = [
        PortConf("d1", DataType.FLOAT, 0, desc="Half-width of first layer (square top)", title="第一层半宽", default=2135.0),
        PortConf("d2", DataType.FLOAT, 0, desc="Width of second layer (octagonal section)", title="第二层宽度", default=5335.0),
        PortConf("d3", DataType.FLOAT, 0, desc="Width of third layer (dodecagonal section)", title="第三层宽度", default=7470.0),
        PortConf("d4", DataType.FLOAT, 0, desc="Width of fourth layer (bottom support)", title="第四层宽度", default=9605.0),
        PortConf("r2", DataType.FLOAT, 0, desc="Radius of second layer (octagonal section)", title="第二层半径", default=4265.0),
        PortConf("r3", DataType.FLOAT, 0, desc="Radius of third layer (dodecagonal section)", title="第三层半径", default=6400.0),
        PortConf("r4", DataType.FLOAT, 0, desc="Radius of fourth layer (bottom support)", title="第四层半径", default=8535.0),
        PortConf("l3", DataType.FLOAT, 0, desc="Height of third segment (total dodecagonal height)", title="第三段高度", default=43890.0),
        PortConf("l2", DataType.FLOAT, 0, desc="Height of second segment (octagonal top height), default l3+29260", title="第二段高度", default=None),
        PortConf("l1", DataType.FLOAT, 0, desc="Height of first segment (square top height), default l2+21950", title="第一段高度", default=None),
        PortConf("A", DataType.FLOAT, 0, desc="Cross-sectional area of truss members [mm²]", title="截面面积", default=4.0),
        PortConf("E", DataType.FLOAT, 0, desc="Young's modulus", title="弹性模量", default=2.1e5),
        PortConf("nu", DataType.FLOAT, 0, desc="Poisson's ratio", title="泊松比", default=0.3),
        PortConf("fx", DataType.FLOAT, 0, desc="Load in x-direction at top nodes [N]", title="X向载荷", default=0.0),
        PortConf("fy", DataType.FLOAT, 0, desc="Load in y-direction at top nodes [N]", title="Y向载荷", default=400.0),
        PortConf("fz", DataType.FLOAT, 0, desc="Load in z-direction at top nodes [N]", title="Z向载荷", default=-100.0)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import bm
        from fealpy.csm.mesh.bar942 import Bar942
        from fealpy.mesh import EdgeMesh
        
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


class TrussTowerGeometry(CNodeType):
    r"""Truss tower geometry data generator.

    The tower body has a rectangular cross-section and is divided into multiple panels along
    the height direction, with optional face diagonal bracing.
    
    Inputs:
        n_panel (int): Number of panels along the z-direction (≥1).
        Lz (float): Total height of the truss tower in z-direction [m].
        Wx (float): Half-width of rectangular cross-section in x-direction [m].
        Wy (float): Half-width of rectangular cross-section in y-direction [m].
        lc (float): Characteristic geometric length for mesh size control [m].
        ne_per_bar (int): Number of elements per bar along length (≥1).
        face_diag (bool): Whether to add in-plane diagonal bracing on four faces.
        
    Outputs:
        node (tensor): Node coordinate array.
        cell (tensor): Cell connectivity array.
    """
    TITLE: str = "桁架塔几何"
    PATH: str = "examples.CSM"
    INPUT_SLOTS = [
        PortConf("n_panel", DataType.INT, 0, desc="Number of panels along z-direction (≥1)", title="面板数量", default=19),
        PortConf("Lz", DataType.FLOAT, 0, desc="Total height of truss tower in z-direction", title="总高度", default=19.0),
        PortConf("Wx", DataType.FLOAT, 0, desc="Half-width of rectangular cross-section in x-direction", title="X向半宽", default=0.45),
        PortConf("Wy", DataType.FLOAT, 0, desc="Half-width of rectangular cross-section in y-direction", title="Y向半宽", default=0.40),
        PortConf("lc", DataType.FLOAT, 0, desc="Characteristic geometric length for mesh size control", title="特征长度", default=0.1),
        PortConf("ne_per_bar", DataType.INT, 0, desc="Number of elements per bar along length (≥1)", title="杆件单元数", default=1),
        PortConf("face_diag", DataType.BOOL, 0, desc="Whether to add in-plane diagonal bracing on four faces", title="面内对角加劲", default=True)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("node", DataType.TENSOR, desc="Node coordinate array", title="节点坐标"),
        PortConf("cell", DataType.TENSOR, desc="Cell connectivity array", title="单元")
    ]

    @staticmethod
    def run(**options):
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
        
        return node, cell