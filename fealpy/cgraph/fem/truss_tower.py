from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["TrussTower"]


class TrussTower(CNodeType):
    r"""Truss Tower Finite Element Model Node.
    
    Inputs:
        space (Space): Lagrange function space for the truss tower model.
        E (float): Young's modulus of the bar elements (Pa).
        nu (float): Poisson's ratio of the bar elements.
        A (Tensor): Cross-sectional areas of all bar elements.
        I (Tensor): Area moments of inertia of all bar elements.
        is_vertical (Tensor): Boolean flags indicating vertical columns (NC,).
        external_load (Tensor): External load vector (NDOF,).
        dirichlet_idx (Tensor): Indices of Dirichlet boundary DOFs.
        
    Outputs:
        K (LinearOperator): Global stiffness matrix with boundary conditions applied.
        F (Tensor): Global load vector with boundary conditions applied.
    
    """
    TITLE: str = "桁架塔有限元模型"
    PATH: str = "simulation.discretization"
    DESC: str = "组装桁架塔结构的刚度矩阵和载荷向量"
    INPUT_SLOTS = [
        PortConf("space", DataType.SPACE, 1, desc="拉格朗日函数空间", title="标量函数空间"),
        PortConf("E", DataType.FLOAT, 1, desc="杆件的弹性模量（Pa）",  title="弹性模量"),
        PortConf("nu", DataType.FLOAT, 1, desc="杆件的泊松比",  title="泊松比"),
        PortConf("load", DataType.FUNCTION, 1, desc="载荷向量", title="外部载荷"),
        PortConf("I1", DataType.FLOAT, 1, desc="深度方向的结构惯性矩", title="结构惯性矩I1"),
        PortConf("I2", DataType.FLOAT, 1, desc="宽度方向的结构惯性矩", title="结构惯性矩I2"),
        
        PortConf("vertical", DataType.INT, 0, desc="竖向杆件的个数",  title="竖向杆件", default=76),
        PortConf("other", DataType.INT, 0, desc="其他杆件的个数",  title="其他杆件", default=176)
    ]
    OUTPUT_SLOTS = [
        PortConf("K", DataType.LINOPS, desc="全局刚度矩阵", title="刚度矩阵",),
        PortConf("F", DataType.TENSOR, desc="全局载荷向量",  title="载荷向量"),
    ]

    @staticmethod
    def run(**options):
        
        from fealpy.backend import bm
        from fealpy.csm.model.truss.truss_tower_data_3d import TrussTowerData3D
        from fealpy.csm.material import BarMaterial
        from fealpy.csm.fem.bar_integrator import BarIntegrator
        
        model = TrussTowerData3D()
        
        material = BarMaterial(
            model=model,
            name="bar",
            elastic_modulus=options.get("E"),
            poisson_ratio=options.get("mu")
        )
        
        space = options.get("space")
        load = options.get("load")
        
        NDOF  = space.number_of_global_dofs()
        K = bm.zeros((NDOF, NDOF), dtype=bm.float64)
        
        # 获取立柱和斜杆的索引
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

        F = model.external_load(load_total=load)
        
        return K, F