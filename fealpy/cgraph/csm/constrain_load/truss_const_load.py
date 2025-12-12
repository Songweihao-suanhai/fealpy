from ...nodetype import CNodeType, PortConf, DataType

__all__ = ["Bar25Load",  "Bar25Boundary",
           "Bar942Load", "Bar942Boundary", 
           "TrussTowerLoad", "TrussTowerBoundary"]


class Bar25Load(CNodeType):
    r"""25-bar truss standard load configuration.

    Applies concentrated force [0, 900, 0] at each top node (z = 5080).
    This is the standard loading for the 25-bar truss benchmark problem.
    
    Inputs:
        mesh (mesh): Mesh object (needed for global vector size).
        
    Outputs:
        external_load (tensor): Global load vector [N].
    """
    TITLE: str = "25杆载荷"
    PATH: str = "examples.CSM"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, 
                 desc="网格对象 (用于确定全局向量大小)", 
                 title="网格")
    ]
    
    OUTPUT_SLOTS = [
        PortConf("external_load", DataType.TENSOR, 
                 desc="全局载荷向量 (展平为一维)", 
                 title="外部载荷")
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import backend_manager as bm
        mesh = options.get("mesh")
        node = mesh.entity('node')
        NN = node.shape[0]
        GD = 3
        
        F = bm.zeros((NN, GD), dtype=bm.float64)
        
        # Apply [0, 900, 0] at top nodes (z == 5080)
        F[node[..., 2] == 5080] = bm.array([0, 900, 0])

        external_load = F.reshape(-1)  # Convert to 1D array
        return external_load

class Bar25Boundary(CNodeType):
    r"""25-bar truss standard boundary conditions.

    Applies fixed support at the bottom four nodes, constraining all translational 
    displacements (X, Y, Z directions).
    
    The bottom nodes are identified by their minimum z-coordinate value.
    
    Inputs:
        mesh (mesh): Mesh object containing node coordinates.
        
    Outputs:
        is_bd_dof (tensor): Boolean array indicating boundary DOFs (all fixed DOFs = True).
    """
    TITLE: str = "25杆边界"
    PATH: str = "examples.CSM"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.TENSOR, 1, 
                 desc="包含节点坐标的网格对象", 
                 title="网格")
    ]
    
    OUTPUT_SLOTS = [
        PortConf("is_bd_dof", DataType.TENSOR, 
                desc="边界自由度的布尔标识数组", 
                title="边界自由度"),
        PortConf("gd_value", DataType.TENSOR, 
                desc="边界位移约束值 (与边界自由度对应)", 
                title="边界位移值")
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import backend_manager as bm
        
        mesh = options.get("mesh")
        node = mesh.entity('node')
        NN = node.shape[0]
        GD = 3
        
        is_bd_dof = bm.zeros(NN * GD, dtype=bm.bool)
        
        # 找到底部四个节点（z坐标最小的节点）
        z_min = bm.min(node[:, 2])
        bottom_nodes = bm.where(node[:, 2] == z_min)[0]
        
        # 固定这些节点的所有自由度 (X, Y, Z)
        for node_idx in bottom_nodes:
            is_bd_dof[node_idx * GD: (node_idx + 1) * GD] = True
            
        gd_value = bm.zeros(NN * GD, dtype=bm.float64)
        return is_bd_dof, gd_value

class Bar942Load(CNodeType):
    r"""942-bar truss standard load configuration.

    Applies concentrated forces at nodes 0 and 1:
    - Node 0: [0, 400, -100]
    - Node 1: [0, 400, -100]
    
    This is the standard loading for the 942-bar truss benchmark problem.
    
    Inputs:
        mesh (mesh): Mesh object containing node coordinates.
        
    Outputs:
        external_load (tensor): Global load vector [N] (flattened).
    """
    TITLE: str = "942杆载荷"
    PATH: str = "examples.CSM"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.TENSOR, 1, 
                desc="包含节点坐标的网格对象", 
                title="网格")
    ]
    
    OUTPUT_SLOTS = [
        PortConf("external_load", DataType.TENSOR, 
                desc="全局载荷向量 (展平为一维)", 
                title="外部载荷")
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import backend_manager as bm
        
        mesh = options.get("mesh")
        node = mesh.entity('node')
        NN = node.shape[0]
        GD = 3
        
        F = bm.zeros((NN, GD), dtype=bm.float64)
        
        # 在节点 0 和 1 施加载荷
        F[0] = bm.array([0, 400, -100])
        F[1] = bm.array([0, 400, -100])
        
        external_load = F.reshape(-1)  # Convert to 1D array
        return external_load
    
    
class Bar942Boundary(CNodeType):
    r"""942-bar truss standard boundary conditions.

    Applies fixed support at nodes 232-243 (0-indexed: 231-242), constraining 
    all degrees of freedom to zero.
    
    Note:
        Node indices in the description (232-243) are 1-indexed,
        but in code we use 0-indexed (231-242).
    
    Inputs:
        mesh (mesh): Mesh object containing node coordinates.
        
    Outputs:
        is_bd_dof (tensor): Boolean array indicating boundary DOFs (all fixed DOFs = True).
    """
    TITLE: str = "942杆边界"
    PATH: str = "examples.CSM"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.TENSOR, 1, 
                 desc="包含节点坐标的网格对象", 
                 title="网格")
    ]
    
    OUTPUT_SLOTS = [
        PortConf("is_bd_dof", DataType.TENSOR, 
                desc="边界自由度的布尔标识数组", 
                title="边界自由度"),
        PortConf("gd_value", DataType.TENSOR, 
                desc="边界位移约束值 (与边界自由度对应)", 
                title="边界位移值")
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import backend_manager as bm
        
        mesh = options.get("mesh")
        node = mesh.entity('node')
        NN = node.shape[0]
        GD = 3
        
        is_bd_dof = bm.zeros(NN * GD, dtype=bm.bool)
        
        # 约束节点232-243的所有自由度
        for i in range(12):
                node_idx = i + 232  # 对应节点233-244
                is_bd_dof[node_idx * GD : (node_idx + 1) * GD] = True
                
        gd_value = bm.zeros(NN * GD, dtype=bm.float64)
        return is_bd_dof, gd_value


class TrussTowerLoad(CNodeType):
    r"""Truss tower top load applicator.

    Applies total vertical load uniformly distributed at the top nodes (z > 18.999).
    Each top node receives an equal fraction of the total load in the negative z-direction.
    
    Inputs:
        mesh (mesh): Mesh object containing node coordinates.
        load_total (float): Total vertical load applied at tower top [N].
        
    Outputs:
        external_load (tensor): Global load vector [N] (flattened).
    """
    TITLE: str = "桁架塔载荷"
    PATH: str = "examples.CSM"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.TENSOR, 1, 
                 desc="包含节点坐标的网格对象", 
                 title="网格"),
        PortConf("load_total", DataType.FLOAT, 0, 
                 desc="施加在塔顶的总竖向载荷 [N]", 
                 title="总载荷", 
                 default=84820.0)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("external_load", DataType.TENSOR, 
                 desc="全局载荷向量 (展平为一维)", 
                 title="外部载荷")
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import backend_manager as bm
        
        mesh = options.get("mesh")
        node = mesh.entity('node')
        NN = node.shape[0]
        dofs_per_node = 3
        load_total = options.get("load_total")
        
        # 直接创建一维载荷向量 (NN*3,)
        external_load = bm.zeros((NN * dofs_per_node,), dtype=bm.float64)
        
        # 找到顶部节点 (z > 18.999)
        top_nodes = bm.where(node[:, 2] > 18.999)[0]
        
        # 将总载荷均匀分配到每个顶部节点
        load_per_node = load_total / len(top_nodes)
        for i in top_nodes:
            external_load[3*i + 2] = -load_per_node  # Z方向向下（负载荷）

        return external_load


class TrussTowerBoundary(CNodeType):
    r"""Truss tower standard boundary conditions.

    Applies fixed support at all bottom nodes (z ≈ 0), constraining all 
    translational displacements.
    
    Bottom nodes are identified by having z-coordinate less than 0.001.
    
    Inputs:
        mesh (mesh): Mesh object containing node coordinates.
        
    Outputs:
        is_bd_dof (tensor): Boolean array indicating boundary DOFs (all fixed DOFs = True).
    """
    TITLE: str = "桁架塔边界"
    PATH: str = "examples.CSM"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.TENSOR, 1, 
                 desc="包含节点坐标的网格对象", 
                 title="网格")
    ]
    
    OUTPUT_SLOTS = [
        PortConf("is_bd_dof", DataType.TENSOR, 
                desc="边界自由度的布尔标识数组", 
                title="边界自由度"),
        PortConf("gd_value", DataType.TENSOR, 
                desc="边界位移约束值 (与边界自由度对应)", 
                title="边界位移值")
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import backend_manager as bm
        
        mesh = options.get("mesh")
        node = mesh.entity('node')
        NN = node.shape[0]
        GD = 3

        is_bd_dof = bm.zeros(NN * GD, dtype=bm.bool)

        # 找到底部所有节点 (z ≈ 0)
        bottom_nodes = bm.where(node[:, 2] < 1e-6)[0]
        
        # 固定这些节点的所有自由度
        for node_idx in bottom_nodes:
            is_bd_dof[node_idx * GD: (node_idx + 1) * GD] = True

        gd_value = bm.zeros(NN * GD, dtype=bm.float64)
        return is_bd_dof, gd_value