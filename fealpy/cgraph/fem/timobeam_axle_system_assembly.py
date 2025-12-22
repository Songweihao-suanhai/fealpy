from ..nodetype import CNodeType, PortConf, DataType

__all__ = ['TimobeamAxleSystemAssembly']


class TimobeamAxleSystemAssembly(CNodeType):
    r"""Assemble global stiffness matrix and load vector for Timoshenko beam-axle system.
    
    Inputs:
        mesh (MESH): Mesh object containing both beam and axle elements with required celldata:
            - 'type': Element type array (0=beam, 1=axle/spring)
            - Beam elements: 'E', 'G', 'Ax', 'Ay', 'Az', 'Iy', 'Iz', 'J'
            - Axle elements: 'k_axle'
            
    Outputs:
        K (LINOPS): Global stiffness matrix in CSR format (6*NN, 6*NN).
        
    Note:
        - Automatically identifies beam elements (type==0) and axle elements (type==1)
        - Each node has 6 DOFs: 3 translational (ux, uy, uz) and 3 rotational (θx, θy, θz)
        - K = K_beam + K_axle
    """
    TITLE: str = "铁木辛柯梁-轴系统组装"
    PATH: str = "simulation.discretization"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, desc="包含梁和轴单元及属性的网格对象", title="网格")
    ]

    OUTPUT_SLOTS = [
       PortConf("K", DataType.LINOPS, desc="全局刚度矩阵 (稀疏CSR格式)", title="全局刚度矩阵")
    ]
    
    
    @staticmethod
    def run(**options):
        from fealpy.backend import bm
        from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
        from fealpy.fem import BilinearForm
        from fealpy.csm.fem.axle_integrator import AxleIntegrator
        from fealpy.csm.fem.timoshenko_beam_integrator import TimoshenkoBeamIntegrator
        
        mesh = options.get("mesh")
        
        if 'type' not in mesh.celldata:
            raise ValueError("Missing required celldata field: 'type' (0=beam, 1=axle)")
        
        beam_indices = bm.where(mesh.celldata['type'] == 0)[0]
        axle_indices = bm.where(mesh.celldata['type'] == 1)[0]
        
        if len(beam_indices) == 0:
            raise ValueError("No beam elements found in mesh (type == 0)")
        if len(axle_indices) == 0:
            raise ValueError("No axle elements found in mesh (type == 1)")
        
        space = LagrangeFESpace(mesh, p=1)
        tspace = TensorFunctionSpace(space, shape=(-1, 6))
        
        bform = BilinearForm(tspace)
        
        # === 组装梁单元刚度矩阵 ===
        class BeamModel:
            def __init__(self):
                self.GD = 3
                self.Ax = mesh.celldata['Ax'][beam_indices]
                self.Ay = mesh.celldata['Ay'][beam_indices]
                self.Az = mesh.celldata['Az'][beam_indices]
                self.Iy = mesh.celldata['Iy'][beam_indices]
                self.Iz = mesh.celldata['Iz'][beam_indices]
                self.J = mesh.celldata['J'][beam_indices]
                self.FSY = mesh.celldata['shear_factor'][beam_indices]
                self.FSZ = mesh.celldata['shear_factor'][beam_indices]
    
        class BeamMaterial:
            def __init__(self):
                self.E = mesh.celldata['E'][beam_indices]
                self.mu = mesh.celldata['G'][beam_indices]
        
        beam_model = BeamModel()
        beam_material = BeamMaterial()
        
        beam_integrator = TimoshenkoBeamIntegrator(
            space=tspace,
            model=beam_model,
            material=beam_material,
            index=beam_indices  # 仅对梁单元进行积分
        )
        bform.add_integrator(beam_integrator)
        
        # === 组装轴单元刚度矩阵 ===
        # 获取轴刚度系数（取第一个轴单元的值作为标量）
        k_axle_array = mesh.celldata['k_axle'][axle_indices]
        k_axle_scalar = float(k_axle_array[0])
        
        
        class AxleModel:
            def __init__(self):
                self.GD = 3
        
        class AxleMaterial:
            def __init__(self):
                self.k_axle = k_axle_scalar
        
        axle_model = AxleModel()
        axle_material = AxleMaterial()
        
        axle_integrator = AxleIntegrator(
            space=tspace,
            model=axle_model,
            material=axle_material,
            index=axle_indices  # 仅对轴单元进行积分
        )
        bform.add_integrator(axle_integrator)
        K = bform.assembly()
        
        return K
