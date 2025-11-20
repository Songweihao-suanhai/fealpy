from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["ChannelBeam"]


class ChannelBeam(CNodeType):
    r"""Channel Beam Material Definition Node.

    Inputs:
        mu_y (FLOAT): Ratio of maximum to average shear stress for y-direction shear.
        mu_z (FLOAT): Ratio of maximum to average shear stress for z-direction shear.
        GD (INT): Geometric dimension of the model.
        space (SPACE): Lagrange function space.
        beam_E (FLOAT): Elastic modulus of the beam material.
        beam_nu (FLOAT): Poisson’s ratio of the beam material.
        beam_density (FLOAT): Density of the beam material.
        load_case (MENU): Load case selection.
        gravity (FLOAT): Gravitational acceleration. Default 9.81.
        dirichlet_dof (FUNCTION): Returns Dirichlet degrees of freedom.

    Outputs:
        K (tensor): Global stiffness matrix after applying boundary constraints.
        F (tensor): Global load vector after applying the selected load type and boundary conditions.
    """
    TITLE: str = "槽形梁有限元模型"
    PATH: str = "simulation.discretization"
    DESC: str = """"该节点基于前处理阶段给定的几何参数与材料参数，构建槽形梁的有限元模型。
            调用对应材料模型计算单元刚度矩阵。随后，将所有单元刚度汇总组装为全局刚度矩阵，
            并施加 Dirichlet 边界条件，输出包含边界约束后的全局刚度矩阵与全局载荷向量。"""
    
    INPUT_SLOTS = [
        PortConf("mu_y", DataType.FLOAT, 1, desc="y方向剪切应力的最大值与平均值比例因子", 
                 title="y向剪切因子"),
        PortConf("mu_z", DataType.FLOAT, 1, desc="z方向剪切应力的最大值与平均值比例因子", 
                 title="z向剪切因子"),
        PortConf("GD", DataType.INT, 1, desc="模型的几何维数", title="几何维数"),
        PortConf("space", DataType.SPACE, 1, desc="拉格朗日函数空间", title="标量函数空间"),
        PortConf("beam_E", DataType.FLOAT, 1, desc="梁的弹性模量", title="梁的弹性模量"),
        PortConf("beam_nu", DataType.FLOAT, 1, desc="梁的泊松比", title="梁的泊松比"),
        PortConf("beam_density", DataType.FLOAT, 1, desc="梁的密度", title="梁的密度"),
        PortConf("load_case", DataType.MENU, 1, desc="载荷工况选择", title="载荷工况"),
        PortConf("dirichlet_dof", DataType.FUNCTION, 1, desc="返回 Dirichlet 自由度", title="边界自由度"),
        PortConf("gravity", DataType.FLOAT, 0, desc="重力加速度", title="重力加速度", default=9.81),
        
    ]
    
    OUTPUT_SLOTS = [
        PortConf("K", DataType.LINOPS, desc="含边界条件处理后的刚度矩阵", title="全局刚度矩阵",),
        PortConf("F", DataType.TENSOR, desc="含边界条件作用的全局载荷向量",  title="全局载荷向量"),
    ]
    
    @staticmethod
    def run(**options):
        
        from fealpy.backend import bm
        from fealpy.csm.model.beam.channel_beam_data_3d import ChannelBeamData3D
        from fealpy.functionspace import TensorFunctionSpace
        from fealpy.csm.material import TimoshenkoBeamMaterial
        from fealpy.csm.fem.timoshenko_beam_integrator import TimoshenkoBeamIntegrator
        from fealpy.fem import (
                BilinearForm,  
                LinearForm,
                VectorSourceIntegrator,
                DirichletBC
                )

        model = ChannelBeamData3D(mu_y=options.get("mu_y"), mu_z=options.get("mu_z"))
        
        GD = options.get("GD")
        space = options.get("space")
        load_case = options.get("load_case")
        g = options.get("gravity")
        dirichlet_dof = options.get("dirichlet_dof")
        
        beam_E  = options.get("beam_E")
        beam_nu = options.get("beam_nu")
        rho = options.get("beam_density")
        material = TimoshenkoBeamMaterial(model=model, 
                                        name="Timoshenko_beam",
                                        elastic_modulus=beam_E,
                                        poisson_ratio=beam_nu,
                                        density=rho)
        
        tspace = TensorFunctionSpace(space, shape=(-1, GD*2))
        bform = BilinearForm(tspace)
        bform.add_integrator(TimoshenkoBeamIntegrator(space=tspace, 
                                        model=model, 
                                        material=material))
        K = bform.assembly()
        
        mesh = tspace.mesh
        gdof = tspace.number_of_global_dofs()
        node = mesh.entity('node')
        
        F = bm.zeros(gdof, dtype=bm.float64)
        
        if load_case == 1:
                tip_node_idx = bm.argmax(node[:, 0])  # 最右端节点
                load = model.tip_load(load_case=1)
                for i in range(6):
                        F[tip_node_idx * 6 + i] = load[i]
        elif load_case == 2:
                q_z = - model.Ax * rho * g 
                NC = mesh.number_of_cells() 
                shape=(NC, 4) +(6, )              
                load = bm.zeros(shape, dtype=bm.float64)
                load[..., 2] = q_z  
                lform = LinearForm(tspace)
                lform.add_integrator(VectorSourceIntegrator(
                        load, q=4))
                F = lform.assembly()

        threshold = bm.zeros(gdof, dtype=bool)
        fixed_dofs = dirichlet_dof
        threshold[fixed_dofs] = True

        bc = DirichletBC(
                space=tspace,
                gd=model.dirichlet,
                threshold=threshold
        )
        K, F = bc.apply(K, F)

        return K, F