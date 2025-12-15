from ..nodetype import CNodeType, PortConf, DataType

__all__ = ['AssembleBarStiffness',
           'DirichletMethodBC',
           'PenaltyMethodBC']


class AssembleBarStiffness(CNodeType):
    """Assemble global stiffness matrix for bar/truss elements.
    
    Inputs:
        mesh (EdgeMesh): Mesh object containing bar elements with celldata['A'] and celldata['E'].
        
    Outputs:
        K (sparse matrix): Global stiffness matrix in CSR format (GD*NN, GD*NN).
    
    Note:
        mesh.celldata['A']: Cross-sectional area of each bar element.
        mesh.celldata['E']: Young's modulus of each bar element.
    """
        
    TITLE: str = "杆单元刚度矩阵组装"
    PATH: str = "simulation.discretization"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, 
                 desc="包含杆单元的网格对象", 
                 title="网格")
    ]
    
    OUTPUT_SLOTS = [
        PortConf("K", DataType.LINOPS, 
                 desc="全局刚度矩阵 (稀疏CSR格式)", 
                 title="刚度矩阵")
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import backend_manager as bm
        from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
        from fealpy.fem import BilinearForm
        from fealpy.csm.fem.bar_integrator import BarIntegrator
        
        mesh = options.get("mesh")
        # 简单的 PDE 模型类
        class SimpleBarModel:
            def __init__(self):
                self.GD = 3
                self.A = mesh.celldata['A']
        
        # 简单的材料类（只存储 E）
        class SimpleMaterial:
            def __init__(self):
                self.E = mesh.celldata['E']
        
        
        # 创建模型和材料对象
        model = SimpleBarModel()
        material = SimpleMaterial()
        
        space = LagrangeFESpace(mesh, p=1)
        tspace = TensorFunctionSpace(space, shape=(-1, 3))
        
        bform = BilinearForm(tspace)
        integrator = BarIntegrator(space=tspace, model=model, material=material)
        bform.add_integrator(integrator)
        K = bform.assembly()
        return K


class DirichletMethodBC(CNodeType):
    r"""Apply Dirichlet boundary conditions using Direct Method.

    Directly modifies the stiffness matrix and load vector by replacing
    constrained rows/columns with identity matrix structure.
    
    Inputs:
        mesh (EdgeMesh): Mesh object containing bar elements.
        K (linops): Global stiffness matrix in CSR format.
        
    Outputs:
        K_bc (linops): Modified stiffness matrix with boundary conditions applied.
        F_bc (tensor): Modified load vector with boundary conditions applied.
    
    Note:
        - mesh.nodedata['load']: Applied loads at nodes (NN, 3).
        - mesh.nodedata['constraint']: Constraint flags (NN, 4), format [node_idx, flag_x, flag_y, flag_z].
        - This method maintains better numerical conditioning than Penalty Method.
        - Constrained DOF displacements are set to 0 (can be extended for non-zero prescribed values).
    """
    TITLE: str = "迪利克雷边界条件处理"
    PATH: str = "simulation.boundary"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, 
                 desc="包含杆单元的网格对象",  
                 title="网格"),
        PortConf("K", DataType.LINOPS, 1, 
                 desc="全局刚度矩阵 (稀疏CSR格式)", 
                 title="刚度矩阵")
    ]
    
    OUTPUT_SLOTS = [
        PortConf("K_bc", DataType.LINOPS, 
                 desc="应用边界条件后的刚度矩阵", 
                 title="边界处理后的刚度矩阵"),
        PortConf("F_bc", DataType.TENSOR, 
                 desc="应用边界条件后的载荷向量", 
                 title="边界处理后的载荷向量")
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import backend_manager as bm
        from fealpy.sparse import CSRTensor
        
        # 获取输入
        mesh = options.get("mesh")
        K = options.get("K")
        
        # 验证必需数据
        if not hasattr(mesh, 'nodedata'):
            raise ValueError("Mesh must have nodedata attribute")
        if 'load' not in mesh.nodedata:
            raise ValueError("Mesh nodedata must contain 'load' array")
        if 'constraint' not in mesh.nodedata:
            raise ValueError("Mesh nodedata must contain 'constraint' array")
        
        NN = mesh.number_of_nodes()
        load = mesh.nodedata['load'] 
        constraint = mesh.nodedata['constraint'] 
        
        # 初始化载荷向量和刚度矩阵
        F_bc = load.flatten()
        K_dense = K.toarray()
    
        # 提取约束信息
        node_indices = constraint[:, 0].astype(bm.int32)
        constraint_flags = constraint[:, 1:4]
        
        # 构建节点自由度映射
        node_dofs = bm.zeros((NN, 3), dtype=bm.int32)
        for j in range(3):
            node_dofs[:, j] = 3 * node_indices + j

        # 找出所有被约束的自由度
        is_constrained = constraint_flags > 0.5
        constrained_dofs = node_dofs[is_constrained]
        
        # 迪利克雷边界条件处理
        K_dense[constrained_dofs, :] = 0.0
        K_dense[:, constrained_dofs] = 0.0
        K_dense[constrained_dofs, constrained_dofs] = 1.0
        
        F_bc[constrained_dofs] = 0.0
        
        # 转换回 CSR 稀疏格式
        rows, cols = bm.nonzero(K_dense)
        values = K_dense[rows, cols]
        
        crow = bm.zeros(K_dense.shape[0] + 1, dtype=bm.int64)
        for i in range(len(rows)):
            crow[rows[i] + 1] += 1
        crow = bm.cumsum(crow)
        
        K_bc = CSRTensor(crow, cols, values, spshape=K_dense.shape)
        
        return K_bc, F_bc

class PenaltyMethodBC(CNodeType):
    r"""Apply Dirichlet boundary conditions using Penalty Method.

    Inputs:
        mesh (EdgeMesh): Mesh object containing bar elements.
        K (linops): Global stiffness matrix in CSR format.
        penalty (float, optional): Penalty factor (default: 1e12).
        
    Outputs:
        K_bc (linops): Modified stiffness matrix with boundary conditions applied.
        F_bc (tensor): Modified load vector with boundary conditions applied.
    """
    TITLE: str = "乘大数法边界条件处理"
    PATH: str = "simulation.boundary"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, 
                 desc="包含杆单元的网格对象",  
                 title="网格"),
        PortConf("K", DataType.LINOPS, 1, 
                 desc="全局刚度矩阵 (稀疏CSR格式)", 
                 title="刚度矩阵"),
        PortConf("penalty", DataType.FLOAT, 0, 
                 desc="惩罚系数 (建议值: 1e12)", 
                 title="惩罚系数", 
                 default=1e12)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("K_bc", DataType.LINOPS, 
                 desc="应用边界条件后的刚度矩阵", 
                 title="边界处理后的刚度矩阵"),
        PortConf("F_bc", DataType.TENSOR, 
                 desc="应用边界条件后的载荷向量", 
                 title="边界处理后的载荷向量")
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import backend_manager as bm
        from fealpy.sparse import CSRTensor
        
        # 获取输入
        mesh = options.get("mesh")
        K = options.get("K")
        penalty = options.get("penalty", 1e12)
        
        # 验证必需数据
        if not hasattr(mesh, 'nodedata'):
            raise ValueError("Mesh must have nodedata attribute")
        if 'load' not in mesh.nodedata:
            raise ValueError("Mesh nodedata must contain 'load' array")
        if 'constraint' not in mesh.nodedata:
            raise ValueError("Mesh nodedata must contain 'constraint' array")
        
        NN = mesh.number_of_nodes()
        load = mesh.nodedata['load'] 
        constraint = mesh.nodedata['constraint'] 
        
        F_bc = load.flatten()
        K_dense = K.toarray()
    
        node_indices = constraint[:, 0].astype(bm.int32)
        constraint_flags = constraint[:, 1:4]
        
        node_dofs = bm.zeros((NN, 3), dtype=bm.int32)
        for j in range(3):
            node_dofs[:, j] = 3 * node_indices + j

        is_constrained = constraint_flags > 0.5
        constrained_dofs = node_dofs[is_constrained]
        
        K_dense[constrained_dofs, constrained_dofs] *= penalty
        F_bc[constrained_dofs] = 0.0 * K_dense[constrained_dofs, constrained_dofs]

        # 转换回 CSR 稀疏格式
        rows, cols = bm.nonzero(K_dense)
        values = K_dense[rows, cols]
        
        crow = bm.zeros(K_dense.shape[0] + 1, dtype=bm.int64)
        for i in range(len(rows)):
            crow[rows[i] + 1] += 1
        crow = bm.cumsum(crow)
        
        K_bc = CSRTensor(crow, cols, values, spshape=K_dense.shape)
        
        return K_bc, F_bc
    

class TrussTower(CNodeType):
    r"""Truss Tower Finite Element Model Node.
    
    Inputs:
        dov (float): Outer diameter of vertical rods (m).
        div (float): Inner diameter of vertical rods (m).
        doo (float): Outer diameter of other rods (m).
        dio (float): Inner diameter of other rods (m).
        space_type (str): Type of function space (e.g., "lagrangespace").
        GD (int): Geometric dimension of the model.
        mesh (mesh): A scalar Lagrange function space.
        E (float): Young's modulus of the bar elements (Pa).
        nu (float): Poisson's ratio of the bar elements.
        vertical (INT): Boolean flags indicating vertical columns.
        other (INT): Boolean flags indicating other bars.
        load (float): Total vertical load applied at top nodes.
        dirichlet_dof (Function): Dirichlet boundary DOFs.
        
    Outputs:
        K (LinearOperator): Global stiffness matrix with boundary conditions applied.
        F (Tensor): Global load vector with boundary conditions applied.
    
    """
    TITLE: str = "桁架塔有限元模型"
    PATH: str = "simulation.discretization"
    INPUT_SLOTS = [
        PortConf("dov", DataType.FLOAT, 1,  desc="竖向杆件的外径", title="竖杆外径"),
        PortConf("div", DataType.FLOAT, 1,  desc="竖向杆件的内径", title="竖杆内径"),
        PortConf("doo", DataType.FLOAT, 1,  desc="其他杆件的外径", title="其他杆外径"),
        PortConf("dio", DataType.FLOAT, 1,  desc="其他杆件的内径", title="其他杆内径"),
        PortConf("space_type", DataType.MENU, 0, title="函数空间类型", default="lagrangespace", items=["lagrangespace"]),
        PortConf("GD", DataType.INT, 1, desc="模型的几何维数", title="几何维数"),
        PortConf("mesh", DataType.MESH, 1, desc="桁架塔网格", title="网格"),
        PortConf("E", DataType.FLOAT, 1, desc="杆件的弹性模量",  title="弹性模量"),
        PortConf("nu", DataType.FLOAT, 1, desc="杆件的泊松比",  title="泊松比"),
        PortConf("load", DataType.TENSOR, 1, desc="全局载荷向量，表示总载荷如何分布到顶部节点", title="外部载荷"),
        PortConf("dirichlet_dof", DataType.FUNCTION, 1, desc="Dirichlet边界条件的自由度索引", title="边界自由度"),
        PortConf("vertical", DataType.INT, 0, desc="竖向杆件的个数",  title="竖向杆件", default=76),
        PortConf("other", DataType.INT, 0, desc="其他杆件的个数",  title="其他杆件", default=176)
    ]
    OUTPUT_SLOTS = [
        PortConf("K", DataType.LINOPS, desc="含边界条件处理后的刚度矩阵", title="全局刚度矩阵",),
        PortConf("F", DataType.TENSOR, desc="含边界条件处理后的载荷向量",  title="载荷向量"),
    ]

    @staticmethod
    def run(**options):
        
        from fealpy.backend import bm
        from fealpy.sparse import CSRTensor
        from fealpy.csm.model.truss.truss_tower_data_3d import TrussTowerData3D
        from fealpy.csm.material import BarMaterial
        from fealpy.csm.fem.bar_integrator import BarIntegrator
        from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
        from fealpy.fem import DirichletBC
        
        model = TrussTowerData3D(
            dov=options.get("dov"),
            div=options.get("div"),
            doo=options.get("doo"),
            dio=options.get("dio")
        )
        
        material = BarMaterial(
            model=model,
            name="bar",
            elastic_modulus=options.get("E"),
            poisson_ratio=options.get("nu")
        )
        GD = options.get("GD")
        mesh = options.get("mesh")
        scalar_space = LagrangeFESpace(mesh, p=1)
        space = TensorFunctionSpace(scalar_space, shape=(-1, GD))
        
        gdof  = space.number_of_global_dofs()
        K = bm.zeros((gdof, gdof), dtype=bm.float64)

        # 获取立柱和斜杆的个数
        vertical = options.get("vertical")
        other = options.get("other")
        
        vertical_indices = bm.arange(0, vertical, dtype=bm.int32)
        other_indices = bm.arange(vertical, vertical + other, dtype=bm.int32)

        vertical_integrator = BarIntegrator(
                space=space,
                model=model,
                material=material,
                index=vertical_indices
            )
        KE_vertical = vertical_integrator.assembly(space) # (NC_v, ldof, ldof)
        ele_dofs_vertical = vertical_integrator.to_global_dof(space)  # (NC_v, ldof)
        
        for i in range(len(ele_dofs_vertical)):
            dof = ele_dofs_vertical[i]
            K[dof[:, None], dof] += KE_vertical[i]
        
        other_integrator = BarIntegrator(
                space=space,
                model=model,
                material=material,
                index=other_indices
            )
        KE_other = other_integrator.assembly(space)  # (NC_o, ldof, ldof)
        ele_dofs_other = other_integrator.to_global_dof(space)  # (NC_o, ldof)
        
        for i in range(len(ele_dofs_other)):
            dof = ele_dofs_other[i]
            K[dof[:, None], dof] += KE_other[i]

        F = options.get("load")

        threshold = bm.zeros(gdof, dtype=bool)
        fixed_dofs = options.get("dirichlet_dof")
        threshold[fixed_dofs] = True
        
        rows, cols = bm.nonzero(K)
        values = K[rows, cols]
        crow = bm.zeros(K.shape[0] + 1, dtype=bm.int64)
        for i in range(len(rows)):
            crow[rows[i] + 1] += 1
        crow = bm.cumsum(crow)

        K_sparse = CSRTensor(crow, cols, values, spshape=K.shape)

        bc = DirichletBC(
                space=space,
                gd=lambda p: bm.zeros(p.shape, dtype=bm.float64),  # 返回与插值点相同形状的零数组
                threshold=threshold
            )
        K, F = bc.apply(K_sparse, F)
        
        return K, F
   