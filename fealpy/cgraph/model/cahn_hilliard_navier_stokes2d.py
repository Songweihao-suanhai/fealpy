from typing import Union, Type
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["CHNSPhysics", "RTIMathmatics"]

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

class CHNSPhysics(CNodeType):

    TITLE: str = "CHNS 物理量定义"
    PATH: str = "preprocess.modeling"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, title="网格"),

        PortConf("phitype", DataType.MENU, 0, title="界面函数空间类型", default="lagrange", 
                                            items=["lagrange", "bernstein", "first_nedelec"]),
        PortConf("phi_p", DataType.INT, 0, title="界面函数空间次数", default=1, min_val=1, max_val=10),

        PortConf("utype", DataType.MENU, 0, title="速度空间类型", default="lagrange", 
                                            items=["lagrange", "bernstein", "first_nedelec"]),
        PortConf("u_p", DataType.INT, 0, title="速度空间次数", default=2, min_val=1, max_val=10),
        PortConf("u_gd", DataType.INT, 0, title="速度空间自由度长度", default=2),

        PortConf("ptype", DataType.MENU, 0, title="压力空间类型", default="lagrange", 
                                            items=["lagrange", "bernstein", "first_nedelec"]), 
        PortConf("p_p", DataType.INT, 0, title="压力空间次数", default=1, min_val=1, max_val=10),
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

        bm.set_backend('pytorch')
        bm.set_default_device('cpu')

        mesh = options.get('mesh')
        
        phitype = options.get('phitype')
        phi_p = options.get('phi_p')
        utype = options.get('utype')
        u_p = options.get('u_p')
        u_gd = options.get('u_gd')
        ptype = options.get('ptype')
        p_p = options.get('p_p')

        phispace_class = get_space_class(phitype)
        phispace = phispace_class(mesh, phi_p)

        element_u = (utype.capitalize(), u_p)
        shape_u = (u_gd, -1)
        uspace = functionspace(mesh, element_u, shape=shape_u)

        pspace_class = get_space_class(ptype)
        pspace = pspace_class(mesh, p=p_p)

        phi = phispace.function()
        u = uspace.function()
        p = pspace.function()

        return phi, u, p


class RTIMathmatics(CNodeType):
    TITLE: str = "RTI 数学模型"
    PATH: str = "examples.CFD"
    INPUT_SLOTS = [
        PortConf("phi", DataType.TENSOR, title="相场"),
        PortConf("u", DataType.TENSOR, title="速度"),
        PortConf("p", DataType.TENSOR, title="压力")
    ]
    OUTPUT_SLOTS = [
        PortConf("mobility", DataType.FLOAT, title="迁移率"),
        PortConf("interface", DataType.FLOAT, title="界面参数"),
        PortConf("free_energy", DataType.FLOAT, title="自由能参数"),
        PortConf("time_derivative", DataType.FUNCTION, title="NS时间项系数"),
        PortConf("convection", DataType.FUNCTION, title="对流项系数"),
        PortConf("pressure", DataType.FLOAT, title="压力项系数"),
        PortConf("viscosity", DataType.NONE, title="粘性项系数"),
        PortConf("source", DataType.FUNCTION, title="源项函数"),
        PortConf("is_boundary", DataType.FUNCTION, title="边界"),
        PortConf("phi0", DataType.TENSOR, title="初始相场1"),
        PortConf("phi1", DataType.TENSOR, title="初始相场2"),
        PortConf("u0", DataType.TENSOR, title="初始速度1"),
        PortConf("u1", DataType.TENSOR, title="初始速度2"),
        PortConf("p0", DataType.TENSOR, title="初始压力")
    ]
    @staticmethod
    def run(phi, u, p):
        from fealpy.backend import backend_manager as bm
        from fealpy.decorator import barycentric, cartesian

        mesh = phi.space.mesh
        box = mesh.box
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
        
        eps = 1e-10

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

        @cartesian
        def is_pressure_boundary(p):
            result = bm.zeros_like(p[..., 0], dtype=bool)
            return result

        @cartesian
        def is_velocity_boundary():
            is_x_boundary = is_ux_boundary
            is_y_boundary = is_uy_boundary
            return (is_x_boundary , is_y_boundary)
        
        @cartesian
        def is_ux_boundary(p):
            tag_up = bm.abs(p[..., 1] - box[3]) < eps
            tag_down = bm.abs(p[..., 1] - box[2]) < eps
            tag_left = bm.abs(p[..., 0] - box[0]) < eps
            tag_right = bm.abs(p[..., 0] - box[1]) < eps
            return tag_up | tag_down | tag_left | tag_right 
        
        @cartesian
        def is_uy_boundary(p):
            tag_up = bm.abs(p[..., 1] - box[3]) < eps
            tag_down = bm.abs(p[..., 1] - box[2]) < eps
            return tag_up | tag_down

        @cartesian
        def is_pressure_boundary():
            return 0
        
        phispace = phi.space
        uspace = u.space    
        pspace = p.space
        is_boundary = (is_ux_boundary, is_uy_boundary)
        phi0 = phispace.interpolate(init_interface)
        phi1 = phispace.interpolate(init_interface)
        u0 = uspace.function()
        u1 = uspace.function()
        p0 = pspace.function()
        
        return (mobility, interface, free_energy, time_derivative, convection, pressure, 
                viscosity, source, is_boundary, phi0, phi1, u0, u1, p0)
    


