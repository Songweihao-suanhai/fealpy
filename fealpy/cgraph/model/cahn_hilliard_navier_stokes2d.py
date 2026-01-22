from typing import Union, Type
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["MultiphaseFlowPhysics", "CHNSMathmatics"]

SPACE_CLASSES = {
    "bernstein": ("bernstein_fe_space", "BernsteinFESpace"),
    "lagrange": ("lagrange_fe_space", "LagrangeFESpace"),
    "first_nedelec": ("first_nedelec_fe_space", "FirstNedelecFESpace")
}
def get_space_class(space_type: str) -> Type:
    import importlib
    m = importlib.import_module(
        f"fealpy.functionspace.{SPACE_CLASSES[space_type][0]}"
    )
    return getattr(m, SPACE_CLASSES[space_type][1])

class MultiphaseFlowPhysics(CNodeType):

    TITLE: str = "两相流物理量定义"
    PATH: str = "examples.CFD"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, title="网格"),

        PortConf("phitype", DataType.MENU, 0, title="界面函数空间类型", default="lagrange", 
                                            items=["lagrange", "bernstein", "first_nedelec"]),
        PortConf("phi_p", DataType.INT, 0, title="界面函数空间次数", default=2, min_val=1, max_val=10),

        PortConf("utype", DataType.MENU, 0, title="速度空间类型", default="lagrange", 
                                            items=["lagrange", "bernstein", "first_nedelec"]),
        PortConf("u_p", DataType.INT, 0, title="速度空间次数", default=2, min_val=1, max_val=10),
        PortConf("u_gd", DataType.INT, 0, title="速度空间自由度长度", default=2),

        PortConf("ptype", DataType.MENU, 0, title="压力空间类型", default="lagrange", 
                                            items=["lagrange", "bernstein", "first_nedelec"]), 
        PortConf("p_p", DataType.INT, 0, title="压力空间次数", default=1, min_val=0, max_val=10),
        PortConf("p_ctype", DataType.MENU, 0, title="压力空间连续性类型", default="D", items=["C", "D"]),
    ]
    OUTPUT_SLOTS = [
        PortConf("phi", DataType.TENSOR, title="相场"),
        PortConf("u", DataType.TENSOR, title="速度"),
        PortConf("p", DataType.TENSOR, title="压力")
    ]

    @staticmethod
    def run(**options) -> Union[object]:
        from fealpy.backend import backend_manager as bm
        from fealpy.functionspace import functionspace

        # bm.set_backend('pytorch')
        # bm.set_default_device('cpu')

        mesh = options.get('mesh')
        
        phitype = options.get('phitype')
        phi_p = options.get('phi_p')
        utype = options.get('utype')
        u_p = options.get('u_p')
        u_gd = options.get('u_gd')
        ptype = options.get('ptype')
        p_p = options.get('p_p')
        p_ctype = options.get('p_ctype')

        phispace_class = get_space_class(phitype)
        phispace = phispace_class(mesh, phi_p)

        element_u = (utype.capitalize(), u_p)
        shape_u = (u_gd, -1)
        uspace = functionspace(mesh, element_u, shape=shape_u)

        pspace_class = get_space_class(ptype)
        pspace = pspace_class(mesh, p=p_p)
        pspace_class = get_space_class(ptype)
        if p_p == 0:
            pspace = pspace_class(mesh, p=0, ctype=p_ctype)
        phi = phispace.function()
        u = uspace.function()
        p = pspace.function()

        return phi, u, p


class CHNSMathmatics(CNodeType):
    TITLE: str = "CHNS 数学模型"
    PATH: str = "examples.CFD"
    INPUT_SLOTS = [
        PortConf("phi", DataType.TENSOR, title="相场"),
        PortConf("u", DataType.TENSOR, title="速度"),
        PortConf("p", DataType.TENSOR, title="压力"),
        PortConf("VariableDensity", DataType.BOOL, 0, default=True, title="是否为变密度多相流模型")
    ]
    OUTPUT_SLOTS = [
        PortConf("equation", DataType.LIST, title="方程"),
        PortConf("boundary_condition", DataType.FUNCTION, title="边界条件"),
        PortConf("is_boundary", DataType.FUNCTION, title="边界"),
        PortConf("x0", DataType.LIST, title="初始值")
    ]
    @staticmethod
    def run(phi, u, p,VariableDensity):
        from fealpy.backend import backend_manager as bm
        from fealpy.decorator import barycentric, cartesian
        if VariableDensity is True:
            mesh = phi.space.mesh
            rho = mesh.rho
            Re = mesh.Re
            Fr = mesh.Fr
            epsilon = mesh.epsilon
            Pe = mesh.Pe

            mobility = 1/Pe
            interface = epsilon ** 2
            free_energy = 1
            time_derivative = rho
            convection = rho
            pressure = 1
            viscosity = 1/Re

            def body_force(phi):
                rh = rho(phi)
                @barycentric
                def body_force(bcs, index):
                    result = rh(bcs, index)
                    result = bm.stack((result, result), axis=-1)
                    result[..., 0] = (1/Fr) * result[..., 0] * 0
                    result[..., 1] = (1/Fr) * result[..., 1] * -1
                    return result
                return body_force
            source = body_force

            equation = [{
                "mobility": mobility,
                "interface": interface,
                "free_energy": free_energy,
                "time_derivative": time_derivative,
                "convection": convection,
                "pressure": pressure,
                "viscosity": viscosity,
                "source": source
            }]

            @cartesian
            def init_interface(p):
                '''
                初始化界面
                '''
                x = p[...,0]
                y = p[...,1]
                val =  bm.tanh((y-2-0.1*bm.cos(bm.pi*2*x))/(bm.sqrt(bm.tensor(2))*epsilon))
                return val
            
            @cartesian
            def velocity_dirichlet(p):
                '''
                边界速度
                '''
                result = bm.zeros_like(p, dtype=bm.float64)
                return result
            
            @cartesian
            def pressure_dirichlet(p):
                '''
                边界压力
                '''
                result = bm.zeros_like(p[..., 0], dtype=bm.float64)
                return result
            
            def boundary_condition():
                return velocity_dirichlet, pressure_dirichlet
            
            @cartesian
            def is_ux_boundary(p):
                tag_up = mesh.is_up_boundary(p)
                tag_down = mesh.is_down_boundary(p)
                tag_left = mesh.is_left_boundary(p)
                tag_right = mesh.is_right_boundary(p)
                return tag_up | tag_down | tag_left | tag_right
            
            @cartesian
            def is_uy_boundary(p):
                tag_up = mesh.is_up_boundary(p)
                tag_down = mesh.is_down_boundary(p)
                return tag_up | tag_down
            
            phispace = phi.space
            uspace = u.space    
            pspace = p.space
            is_boundary = (is_ux_boundary, is_uy_boundary)
            phi0 = phispace.interpolate(init_phi)
            phi1 = phispace.interpolate(init_interface)
            u0 = uspace.function()
            u1 = uspace.function()
            p0 = pspace.function()
            x0 = [phi0, phi1, u0, u1, p0]
        else:
            mesh = phi.space.mesh
            Re = mesh.Re
            gamma = mesh.gamma
            epsilon = mesh.epsilon
            lam = mesh.lam
            L_s = mesh.L_s
            V_s = mesh.V_s
            
            
            @cartesian
            def is_wall_boundary(p):
                return (bm.abs(p[..., 1] - 0.125) < 1e-10) | \
                    (bm.abs(p[..., 1] + 0.125) < 1e-10)
            
            
            @cartesian
            def is_uy_Dirichlet(p):
                return (bm.abs(p[..., 1] - 0.125) < 1e-10) | \
                    (bm.abs(p[..., 1] + 0.125) < 1e-10)
            
            @cartesian
            def init_phi(p):
                x = p[..., 0]
                y = p[..., 1]   
                tagfluid0 = bm.logical_and(x > -0.25, x < 0.25)
                tagfluid1 = bm.logical_not(tagfluid0)
                phi = bm.zeros_like(x)
                phi[tagfluid0] = 1.0
                phi[tagfluid1] = -1.0
                return phi
            @cartesian        
            def u_w(p):
                y = p[..., 1]
                result = bm.zeros_like(p)
                tag_up = (bm.abs(y-0.125)) < 1e-10 
                tag_down = (bm.abs(y+0.125)) < 1e-10
                value = bm.where(tag_down, -0.2, 0) + bm.where(tag_up, 0.2, 0)
                result[..., 0] = value 
                return result
            
            @cartesian
            def p_dirichlet(p):
                return bm.zeros_like(p[..., 0])

            @cartesian
            def is_p_dirichlet( p):
                return bm.zeros_like(p[..., 0], dtype=bool)
            
            def boundary_condition():
                return  p_dirichlet
            
            
            equation = [{
                "Re": Re,
                "gamma": gamma,
                "epsilon": epsilon,
                "lam": lam,
                "L_s": L_s,
                "V_s": V_s,
                "u_w": u_w,
                "init_phi": init_phi,
                "is_uy_Dirichlet": is_uy_Dirichlet
            }]

            is_boundary = is_wall_boundary
            phispace = phi.space
            uspace = u.space    
            pspace = p.space
            phi0 = phispace.interpolate(init_phi)
            phi1 = phispace.interpolate(init_phi)
            u0 = uspace.function()
            u1 = uspace.function()
            p0 = pspace.function()
            x0 = [phi0, phi1, u0, u1, p0]
        return (equation, boundary_condition, is_boundary, x0)
    


