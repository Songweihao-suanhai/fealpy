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
                 desc="包含杆单元的及其它属性的网格对象", 
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
        PortConf("mesh", DataType.MESH, 1, desc="包含梁单元及其它属性的网格对象", title="网格")
    ]
    OUTPUT_SLOTS = [
        PortConf("K", DataType.LINOPS, desc="全局刚度矩阵 (稀疏CSR格式)", title="全局刚度矩阵",)
    ]

    @staticmethod
    def run(**options):
        from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
        from fealpy.fem import BilinearForm
        from fealpy.csm.fem.timoshenko_beam_integrator import TimoshenkoBeamIntegrator
    
        mesh = options.get("mesh")
        # 验证网格数据完整性
        required_fields = ['E', 'G', 'Ax', 'Ay', 'Az', 'Iy', 'Iz', 'J']
        for field in required_fields:
            if field not in mesh.celldata:
                raise ValueError(f"Missing required celldata field: '{field}'")
            
        # 简单的 PDE 模型类
        class SimpleBeamModel:
            def __init__(self):
                self.GD = 3
                self.Ax = mesh.celldata['Ax']
                self.Ay = mesh.celldata['Ay']
                self.Az = mesh.celldata['Az']
                self.Iy = mesh.celldata['Iy']
                self.Iz = mesh.celldata['Iz']
                self.J = mesh.celldata['J']
                
                # 剪切修正系数
                if 'mu_y' in mesh.celldata and 'mu_z' in mesh.celldata:
                    self.FSY = mesh.celldata['mu_y']
                    self.FSZ = mesh.celldata['mu_z']
                elif 'shear_factor' in mesh.celldata:
                    self.FSY = mesh.celldata['shear_factor']
                    self.FSZ = mesh.celldata['shear_factor']
        
        class SimpleBeamMaterial:
            def __init__(self):
                self.E = mesh.celldata['E']
                self.mu = mesh.celldata['G']
                
        model = SimpleBeamModel()
        material = SimpleBeamMaterial()  
        
        space = LagrangeFESpace(mesh, p=1)
        tspace = TensorFunctionSpace(space, shape=(-1, 6))
        
        # 组装刚度矩阵
        bform = BilinearForm(tspace)
        integrator = TimoshenkoBeamIntegrator(
            space=tspace, 
            model=model, 
            material=material
        )
        bform.add_integrator(integrator)
        K = bform.assembly()

        return K