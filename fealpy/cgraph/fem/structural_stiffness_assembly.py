from ..nodetype import CNodeType, PortConf, DataType

__all__ = ['BarStiffnessAssembly',
           'SpringStiffnessAssembly',
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
        
        class SimpleBarModel:
            def __init__(self):
                self.GD = 3
                self.A = mesh.celldata['A']
        
        class SimpleMaterial:
            def __init__(self):
                self.E = mesh.celldata['E']
        
        model = SimpleBarModel()
        material = SimpleMaterial()
        
        space = LagrangeFESpace(mesh, p=1)
        tspace = TensorFunctionSpace(space, shape=(-1, 3))
        
        bform = BilinearForm(tspace)
        integrator = BarIntegrator(space=tspace, model=model, material=material)
        bform.add_integrator(integrator)
        K = bform.assembly()
        return K


class SpringStiffnessAssembly(CNodeType):
    r"""Assemble global stiffness matrix for spring elements.
    
    Inputs:
        mesh (MESH): Mesh object containing spring elements with required celldata.
            
    Outputs:
        K (LINOPS): Global stiffness matrix in CSR format (6*NN, 6*NN).
        
    Note:
        Each node has 6 degrees of freedom: 3 translational (ux, uy, uz) and 
        3 rotational (θx, θy, θz).
    """
    TITLE: str = "弹簧单元刚度矩阵组装"
    PATH: str = "simulation.discretization" 
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, 
                desc="包含弹簧单元的网格对象", 
                title="网格")
    ]
    
    OUTPUT_SLOTS = [
        PortConf("K", DataType.LINOPS, title="全局刚度矩阵")
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.backend import bm
        from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
        from fealpy.fem import BilinearForm
        from fealpy.csm.fem.axle_integrator import AxleIntegrator
        
        mesh = options.get("mesh")
        
        # 假设所有单元都是弹簧
        k_axle_value = mesh.celldata['k_axle']
        k_axle = float(k_axle_value[0])
        
        class SimpleSpringModel:
            def __init__(self):
                self.GD = 3 
                
        class SimpleSpringMaterial:
            def __init__(self):
                self.k_axle = k_axle
                
        model = SimpleSpringModel()
        material = SimpleSpringMaterial()
        
        space = LagrangeFESpace(mesh, p=1)
        tspace = TensorFunctionSpace(space, shape=(-1, 6))
        
        bform = BilinearForm(tspace)
        integrator = AxleIntegrator(
            space=tspace, 
            model=model, 
            material=material
        )
        bform.add_integrator(integrator)
        K = bform.assembly()
        
        return K   
    
    
class TimoshenkoBeamAssembly(CNodeType):
    r"""Assemble global stiffness matrix for Timoshenko beam elements.
    
    Each node has 6 degrees of freedom: 3 translational (ux, uy, uz) and 
    3 rotational (θx, θy, θz).
    
    Inputs:
        mesh (MESH): Mesh object containing beam elements with celldata fields.
        
    Outputs:
        K (LINOPS): Global stiffness matrix in CSR format (6*NN, 6*NN).
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
        from fealpy.backend import bm
        from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
        from fealpy.fem import BilinearForm
        from fealpy.csm.fem.timoshenko_beam_integrator import TimoshenkoBeamIntegrator
    
        mesh = options.get("mesh")
            
        class SimpleBeamModel:
            def __init__(self):
                self.GD = 3
                self.Ax = mesh.celldata['Ax']
                self.Ay = mesh.celldata['Ay']
                self.Az = mesh.celldata['Az']
                self.Iy = mesh.celldata['Iy']
                self.Iz = mesh.celldata['Iz']
                self.J = mesh.celldata['J']
                
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
    
        bform = BilinearForm(tspace)
        integrator = TimoshenkoBeamIntegrator(
            space=tspace, 
            model=model, 
            material=material
        )
        bform.add_integrator(integrator)
        K = bform.assembly()

        return K