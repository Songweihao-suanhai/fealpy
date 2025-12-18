from ..nodetype import CNodeType, PortConf, DataType

__all__ = ['BarStiffnessAssembly',
           'TimoshenkoBeamAssembly']

class BarStiffnessAssembly(CNodeType):
    """Assemble global stiffness matrix for bar/truss elements.
    
    Inputs:
        mesh (MESH): Mesh object containing bar elements with celldata['A'] and celldata['E'].
        
    Outputs:
        K (LINOPS): Global stiffness matrix in CSR format (GD*NN, GD*NN).

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
                 title="全局刚度矩阵")
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
    
    
class TimoshenkoBeamAssembly(CNodeType):
    r"""
    """
    TITLE: str = "铁木辛柯梁单元刚度矩阵组装"
    PATH: str = "simulation.discretization" 
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, desc="包含梁单元的网格对象", title="网格")
    ]
    OUTPUT_SLOTS = [
        PortConf("K", DataType.LINOPS, desc="全局刚度矩阵", title="全局刚度矩阵",)
    ]

    @staticmethod
    def run(**options):
        
        from fealpy.backend import bm
        from fealpy.sparse import COOTensor
        from fealpy.csm.model.beam.timobeam_axle_data_3d import TimobeamAxleData3D
        from fealpy.functionspace import LagrangeFESpace,TensorFunctionSpace
        from fealpy.csm.material import BarMaterial
        from fealpy.csm.material import TimoshenkoBeamMaterial
        from fealpy.csm.fem.axle_integrator import AxleIntegrator
        from fealpy.csm.fem.timoshenko_beam_integrator import TimoshenkoBeamIntegrator

        model = TimobeamAxleData3D(
            beam_para=options.get("beam_para"),
            axle_para=options.get("axle_para")
        )

        beam_E  = options.get("beam_E")
        beam_nu = options.get("beam_nu")
        beam_material = TimoshenkoBeamMaterial(model=model, 
                                        name="Timoshenko_beam",
                                        elastic_modulus=beam_E,
                                        poisson_ratio=beam_nu)

        axle_E  = options.get("axle_E")
        axle_nu = options.get("axle_nu")
        axle_material = BarMaterial(model=model,
                                name="Bar",
                                elastic_modulus=axle_E,
                                poisson_ratio=axle_nu)

        GD = options.get("GD")
        mesh = options.get("mesh")
        space = LagrangeFESpace(mesh, p=1)
        NC = mesh.number_of_cells()
        external_load = options.get("external_load")
        dirichlet_dof = options.get("dirichlet_dof")
        penalty = options.get("penalty")
        
        tspace = TensorFunctionSpace(space, shape=(-1, GD*2))

        Dofs = tspace.number_of_global_dofs()
        K = bm.zeros((Dofs, Dofs), dtype=bm.float64)
        F = bm.zeros(Dofs, dtype=bm.float64)
        
        timo_integrator = TimoshenkoBeamIntegrator(tspace, 
                                    model=model,
                                    material=beam_material, 
                                    index=bm.arange(0, NC-10))
        KE_beam = timo_integrator.assembly(tspace)
        ele_dofs_beam = timo_integrator.to_global_dof(tspace)

        for i, dof in enumerate(ele_dofs_beam):
                K[dof[:, None], dof] += KE_beam[i]

        axle_integrator = AxleIntegrator(tspace, 
                                model=model,
                                material=axle_material,
                                index=bm.arange(NC - 10, NC))
        KE_axle = axle_integrator.assembly(tspace)
        ele_dofs_axle = axle_integrator.to_global_dof(tspace)   

        for i, dof in enumerate(ele_dofs_axle):
                K[dof[:, None], dof] += KE_axle[i]

        penalty = penalty
        fixed_dofs = bm.asarray(dirichlet_dof, dtype=int)

        F = external_load
        F[fixed_dofs] *= penalty
        
        for dof in fixed_dofs:
                K[dof, dof] *= penalty
                
        rows, cols = bm.nonzero(K)
        values = K[rows, cols]
        K = COOTensor(bm.stack([rows, cols], axis=0), values, spshape=K.shape)
        
        return K, F