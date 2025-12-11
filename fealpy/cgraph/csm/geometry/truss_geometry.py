from ...nodetype import CNodeType, PortConf, DataType

__all__ = ["Bar25Geometry", "Bar942Geometry", "TrussTowerGeometry"]


class Bar25Geometry(CNodeType):
    r"""25-bar truss geometry data generator.

    Generates node coordinates and cell connectivity for the classic 25-bar 
    space truss structure. 
    
    Inputs:
        None (uses standard geometry configuration)
        
    Outputs:
        node (tensor): Node coordinate array (10 x 3) [mm].
        cell (tensor): Cell connectivity array (25 x 2).
    """
    TITLE: str = "25杆桁架几何"
    PATH: str = "examples.csm"
    INPUT_SLOTS = []
    
    OUTPUT_SLOTS = [
        PortConf("node", DataType.TENSOR, desc="Node coordinate array (10x3)", title="节点"),
        PortConf("cell", DataType.TENSOR, desc="Cell connectivity array (25x2)", title="单元")
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import bm
        
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
        
        
        return node, cell


class Bar942Geometry(CNodeType):
    r"""942-bar truss geometry data generator.

    Generates the classic 942-bar space truss tower structure with four 
    progressively changing cross-sections:
    - Layer 1: Square top section
    - Layer 2: Octagonal section
    - Layer 3: Dodecagonal section
    - Layer 4: Bottom support layer
    
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
        
    Outputs:
        node (tensor): Node coordinate array.
        cell (tensor): Cell connectivity array.
    """
    TITLE: str = "942杆桁架几何"
    PATH: str = "examples.csm"
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
        PortConf("l1", DataType.FLOAT, 0, desc="Height of first segment (square top height), default l2+21950", title="第一段高度", default=None)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("node", DataType.TENSOR, desc="Node coordinate array", title="节点"),
        PortConf("cell", DataType.TENSOR, desc="Cell connectivity array", title="单元")
    ]

    @staticmethod
    def run(**options):
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
        
        return node, cell


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
    PATH: str = "examples.csm"
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
        PortConf("node", DataType.TENSOR, desc="Node coordinate array", title="节点"),
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