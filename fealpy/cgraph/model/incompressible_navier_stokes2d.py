from typing import Union, Type
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["IncompressibleNS2dbenchmark", "IncompressibleFluidPhysics", "IncompressibleNSMathematics"]


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

class IncompressibleNS2dbenchmark(CNodeType):
    r"""2D unsteady incompressible Navier-Stokes equations problem model.

    Inputs:
        example (int): Example number.
    
    Outputs:
        mu (float): Viscosity coefficient.
        rho (float): Density.
        domain (domain): Computational domain.
        velocity (function): Exact velocity solution.
        pressure (function): Exact pressure solution.
        source (function): Source term.
        velocity_dirichlet (function): Dirichlet boundary condition for velocity.
        pressure_dirichlet (function): Dirichlet boundary condition for pressure.
        is_velocity_boundary (function): Predicate function for velocity boundary regions.
        is_pressure_boundary (function): Predicate function for pressure boundary regions.
    """
    TITLE: str = "二维不可压缩 NS 方程基准算例"
    PATH: str = "preprocess.modeling"
    INPUT_SLOTS = [
        PortConf("example", DataType.MENU, 0, title="例子编号", default=1, items=[i for i in range(1, 3)])
    ]
    OUTPUT_SLOTS = [
        PortConf("mu", DataType.FLOAT, title="粘度系数"),
        PortConf("rho", DataType.FLOAT, title = "密度"),
        PortConf("domain", DataType.LIST, title="求解域"),
        PortConf("velocity", DataType.FUNCTION, title="速度真解"),
        PortConf("pressure", DataType.FUNCTION, title="压力真解"),
        PortConf("source", DataType.FUNCTION, title="源"),
        PortConf("velocity_dirichlet", DataType.FUNCTION, title="速度边界条件"),
        PortConf("pressure_dirichlet", DataType.FUNCTION, title="压力边界条件"),
        PortConf("is_velocity_boundary", DataType.FUNCTION, title="速度边界"),
        PortConf("is_pressure_boundary", DataType.FUNCTION, title="压力边界")
    ]

    @staticmethod
    def run(example) -> Union[object]:
        from fealpy.cfd.model import CFDTestModelManager

        manager = CFDTestModelManager('incompressible_navier_stokes')
        model = manager.get_example(example)
        return (model.mu, model.rho, model.domain()) + tuple(
            getattr(model, name)
            for name in ["velocity", "pressure", "source", "velocity_dirichlet", "pressure_dirichlet", "is_velocity_boundary", "is_pressure_boundary"]
        )
    

class IncompressibleFluidPhysics(CNodeType):
    TITLE: str = "不可压缩流体物理量定义"
    PATH: str = "preprocess.modeling"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, title="网格"),

        PortConf("utype", DataType.MENU, 0, title="速度空间类型", default="lagrange", 
                                            items=["lagrange", "bernstein", "first_nedelec"]),
        PortConf("u_p", DataType.INT, 0, title="速度空间次数", default=2, min_val=1, max_val=10),
        PortConf("u_gd", DataType.INT, 0, title="速度空间自由度长度", default=2),

        PortConf("ptype", DataType.MENU, 0, title="压力空间类型", default="lagrange", 
                                            items=["lagrange", "bernstein", "first_nedelec"]), 
        PortConf("p_p", DataType.INT, 0, title="压力空间次数", default=1, min_val=1, max_val=10),

    ]
    OUTPUT_SLOTS = [
        PortConf("u", DataType.FUNCTION, title="速度"),
        PortConf("p", DataType.FUNCTION, title="压力")
    ]

    @staticmethod
    def run(mesh, utype, u_p, u_gd, ptype, p_p) -> Union[object]:
        from fealpy.functionspace import functionspace

        element_u = (utype.capitalize(), u_p)
        shape_u = (u_gd, -1)
        uspace = functionspace(mesh, element_u, shape=shape_u)

        spaceclass = get_space_class(ptype)
        pspace = spaceclass(mesh, p=p_p)

        u = uspace.function()
        p = pspace.function()

        return u, p


class IncompressibleNSMathematics(CNodeType):
    TITLE: str = "不可压缩 NS 数学模型"
    PATH: str = "preprocess.modeling"
    INPUT_SLOTS = [
        PortConf("u", DataType.TENSOR, 1, title="速度"),
        PortConf("p", DataType.TENSOR, 1, title="压力"),
        PortConf("velocity_boundary", DataType.TEXT, 0, title="速度边界条件"),
        PortConf("pressure_boundary", DataType.FLOAT, 0, title="压力边界条件", default=0.0),
        PortConf("velocity_0", DataType.FLOAT, 0, title="初始速度值", default=0.0),
        PortConf("pressure_0", DataType.FLOAT, 0, title="初始压力值", default=0.0),
    ]
    OUTPUT_SLOTS = [
        PortConf("equation", DataType.LIST, title="方程"),
        PortConf("boundary", DataType.FUNCTION, title="边界条件"),
        PortConf("is_boundary", DataType.FUNCTION, title="边界"),
        PortConf("u0", DataType.TENSOR, title="初始速度"),
        PortConf("p0", DataType.TENSOR, title="初始压力")
    ]
    def run(u, p, velocity_boundary, pressure_boundary, velocity_0, pressure_0):
        from fealpy.backend import backend_manager as bm
        from fealpy.decorator import cartesian
        mesh = u.space.mesh
        mu = mesh.mu
        rho = mesh.rho

        time_derivative = rho
        convection = rho
        pressure = 1.0
        viscosity = mu
        
        @cartesian
        def source(p, t):
            x = p[..., 0]
            y = p[..., 1]
            result = bm.zeros(p.shape, dtype=bm.float64)
            result[..., 0] = 0
            result[..., 1] = 0
            return result
        
        equation = [
            {"time_derivative": time_derivative,
             "convection": convection,
             "pressure": pressure,
             "viscosity": viscosity,
             "source": source
            }
        ]

        @cartesian
        def is_velocity_boundary(p):
            if hasattr(mesh, 'geo') is False:
               return ~mesh.is_outlet_boundary(p) 
            else:
                return ~mesh.geo.is_outlet_boundary(p)
        
        @cartesian
        def is_pressure_boundary(p=None):
            if p is None:
                return 1
            else:
                if hasattr(mesh, 'geo') is False:
                    return mesh.is_outlet_boundary(p) 
                else:
                    return mesh.geo.is_outlet_boundary(p) 
            
        def is_boundary():
            is_u_boundary = is_velocity_boundary
            is_p_boundary = is_pressure_boundary
            return (is_u_boundary, is_p_boundary)
     
        @cartesian
        def u_dirichlet( p):
            def ubd(x, y):
                return eval(
                    velocity_boundary,
                    {
                        "bm": bm,
                        "x": x,
                        "y": y
                    }
                )
            x = p[...,0]
            y = p[...,1]
            u_bd = ubd(x, y)
            value = bm.zeros_like(p)
            value[...,0] = u_bd[0]
            value[...,1] = u_bd[1]
            return value

        @cartesian
        def pressure_dirichlet(p, t):
            x = p[...,0]
            y = p[...,1]
            value = bm.ones_like(x) * pressure_boundary
            return value

        @cartesian
        def velocity_dirichlet(p, t):
            x = p[...,0]
            y = p[...,1]

            if hasattr(mesh, 'geo') is False:
                index = mesh.is_inlet_boundary(p)
            else:
                index = mesh.geo.is_inlet_boundary(p)
            result = bm.zeros_like(p)
            result[index] = u_dirichlet(p[index])
            return result
        
        def dirichlet_boundary():
            u_dirichlet = velocity_dirichlet
            p_dirichlet = pressure_dirichlet
            return (u_dirichlet, p_dirichlet)
        
        def velocity0(p, t):
            x = p[...,0]
            y = p[...,1]
            value = velocity_0 * bm.ones_like(p)
            return value

        def pressure0(p, t):
            x = p[..., 0]
            val = pressure_0 * bm.ones_like(x)
            return val
            
        uspace = u.space
        pspace = p.space
        u0 = uspace.interpolate(cartesian(lambda p:velocity0(p, 0)))
        p0 = pspace.interpolate(cartesian(lambda p:pressure0(p, 0)))

        return (equation, dirichlet_boundary, is_boundary, u0, p0)
    
