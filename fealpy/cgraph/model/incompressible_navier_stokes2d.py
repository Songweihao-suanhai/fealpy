from typing import Union, Type
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["IncompressibleNS2dbenchmark", "IncompressibleCylinder2d",
           "IncompressibleNSPhysics", "IncompressibleNSMathematics"]


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
    

class IncompressibleCylinder2d(CNodeType):
    r"""2D unsteady incompressible Navier-Stokes equations model for flow around a cylinder.

    Inputs:
        mu (float): Viscosity coefficient.
        rho (float): Density.
        cx (float): x-coordinate of the cylinder center.
        cy (float): y-coordinate of the cylinder center.
        radius (float): Radius of the cylinder.
        n_circle (int): Number of discretization points on the circle.
        h (float): Mesh size.
    
    Outputs:
        mu (float): Viscosity coefficient.
        rho (float): Density.
        domain (domain): Computational domain.
        source (function): Source term.
        velocity_0 (function): Initial velocity field.
        pressure_0 (function): Initial pressure field.
        velocity_dirichlet (function): Dirichlet boundary condition for velocity.
        pressure_dirichlet (function): Dirichlet boundary condition for pressure.
        is_velocity_boundary (function): Predicate function for velocity boundary regions.
        is_pressure_boundary (function): Predicate function for pressure boundary regions.
        mesh (mesh): Computational mesh.
    """
    TITLE: str = "二维非稳态圆柱绕流问题模型"
    PATH: str = "preprocess.modeling"
    DESC: str = """该节点建立二维非稳态不可压缩圆柱绕流数值模型，自动生成带圆柱障碍物的计算网格，定义
                入口抛物线速度、出口压力及各类边界条件, 为Navier-Stokes方程求解提供完整物理场输入。"""
    INPUT_SLOTS = [
        PortConf("mu", DataType.FLOAT, 0, title="粘度", default=0.001),
        PortConf("rho", DataType.FLOAT, 0, title="密度", default=1.0),
        PortConf("cx", DataType.FLOAT, 0, title="圆心x坐标", default=0.2),
        PortConf("cy", DataType.FLOAT, 0, title="圆心y坐标", default=0.2),
        PortConf("radius", DataType.FLOAT, 0, title="半径", default=0.05),
        PortConf("n_circle", DataType.INT, 0, title="圆离散点数", default=200),
        PortConf("h", DataType.FLOAT, 0, title="网格尺寸", default=0.05)
    ]
    OUTPUT_SLOTS = [
        PortConf("mu", DataType.FLOAT, title="粘度"),
        PortConf("rho", DataType.FLOAT, title = "密度"),
        PortConf("domain", DataType.LIST, title="求解域"),
        PortConf("source", DataType.FUNCTION, title="源"),
        PortConf("velocity_0", DataType.FUNCTION, title="初始速度"),
        PortConf("pressure_0", DataType.FUNCTION, title="初始压力"),
        PortConf("velocity_dirichlet", DataType.FUNCTION, title="速度边界条件"),
        PortConf("pressure_dirichlet", DataType.FUNCTION, title="压力边界条件"),
        PortConf("is_velocity_boundary", DataType.FUNCTION, title="速度边界"),
        PortConf("is_pressure_boundary", DataType.FUNCTION, title="压力边界"),
        PortConf("mesh", DataType.MESH, title = "网格")
    ]

    @staticmethod
    def run(mu, rho, cx, cy, radius, n_circle, h) -> Union[object]:
        from fealpy.backend import backend_manager as bm
        from fealpy.decorator import cartesian
        from fealpy.backend import TensorLike
        from typing import Sequence
        class PDE:
            def __init__(self, options: dict = None):
                self.options = options
                self.atol = 1e-10
                self.mu = options.get('mu', 0.001)
                self.rho = options.get('rho', 1.0)
                self.box = options.get('box', [0.0, 2.2, 0.0, 0.41])
                self.cx = options.get('cx', 0.2)
                self.cy = options.get('cy', 0.2)
                self.radius = options.get('radius', 0.05)
                self.n_circle = options.get('n_circle', 100)
                self.h = options.get('h', 0.01)
                self.center = (cx, cy)
                self.mesh = self.init_mesh()

            def get_dimension(self) -> int: 
                """Return the geometric dimension of the domain."""
                return 2

            def domain(self) -> Sequence[float]:
                """Return the computational domain [xmin, xmax, ymin, ymax]."""
                return self.box
            
            def init_mesh(self): 
                import gmsh 
                from fealpy.mesh import TriangleMesh 
                box = box 
                center = center 
                radius = radius 
                n_circle = n_circle 
                h = h 
                cx = center[0]
                cy = center[1] 
                gmsh.initialize() 
                gmsh.model.add("rectangle_with_polygon_hole") 
                xmin, xmax, ymin, ymax = box 
                p1 = gmsh.model.geo.addPoint(xmin, ymin, 0) 
                p2 = gmsh.model.geo.addPoint(xmax, ymin, 0) 
                p3 = gmsh.model.geo.addPoint(xmax, ymax, 0) 
                p4 = gmsh.model.geo.addPoint(xmin, ymax, 0) 
                l1 = gmsh.model.geo.addLine(p1, p2) 
                l2 = gmsh.model.geo.addLine(p2, p3) 
                l3 = gmsh.model.geo.addLine(p3, p4) 
                l4 = gmsh.model.geo.addLine(p4, p1) 
                outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4]) 
                theta = bm.linspace(0, 2*bm.pi, n_circle, endpoint=False) 
                circle_pts = [] 
                for t in theta:
                    x = cx + radius * bm.cos(t) 
                    y = cy + radius * bm.sin(t) 
                    pid = gmsh.model.geo.addPoint(x, y, 0) 
                    circle_pts.append(pid) 
                circle_lines = [] 
                for i in range(n_circle): 
                    l = gmsh.model.geo.addLine(circle_pts[i], circle_pts[(i + 1) % n_circle]) 
                    circle_lines.append(l) 
                circle_loop = gmsh.model.geo.addCurveLoop(circle_lines) 
                surf = gmsh.model.geo.addPlaneSurface([outer_loop, circle_loop]) 
                gmsh.model.geo.synchronize() 
                inlet = gmsh.model.addPhysicalGroup(1, [l4], tag = 1) 
                gmsh.model.setPhysicalName(1, 1, "inlet") 
                outlet = gmsh.model.addPhysicalGroup(1, [l2], tag = 2) 
                gmsh.model.setPhysicalName(1, 2, "outlet") 
                wall = gmsh.model.addPhysicalGroup(1, [l1, l3], tag = 3) 
                gmsh.model.setPhysicalName(1, 3, "walls") 
                cyl = gmsh.model.addPhysicalGroup(1, circle_lines, tag = 4) 
                gmsh.model.setPhysicalName(1, 4, "cylinder") 
                domain = gmsh.model.addPhysicalGroup(2, [surf], tag = 5) 
                gmsh.model.setPhysicalName(2, 5, "fluid") 
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h) 
                gmsh.model.mesh.generate(2) 
                node_tags, node_coords, _ = gmsh.model.mesh.getNodes() 
                elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2) 
                tri_nodes = elem_node_tags[0].reshape(-1, 3) - 1 # 转为从0开始索引 
                node_coords = bm.array(node_coords).reshape(-1, 3)[:, :2] 
                tri_nodes = bm.array(tri_nodes, dtype=bm.int32) 
                boundary = [] 
                boundary_tags = [1, 2, 3, 4] 
                for tag in boundary_tags: 
                    node_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, tag) # 转换为从 0 开始的索引 
                    boundary.append(bm.array(node_tags - 1, dtype=bm.int32)) 
                boundary = boundary 
                gmsh.finalize() 
                return TriangleMesh(node_coords, tri_nodes)
            
            @cartesian
            def velocity_dirichlet(self, p:TensorLike, t) -> TensorLike:
                inlet = self.inlet_velocity(p)
                outlet = self.inlet_velocity(p)
                is_inlet = self.is_inlet_boundary(p)
                is_outlet = self.is_outlet_boundary(p)
                
                result = bm.zeros_like(p, dtype=p.dtype)
                result[is_inlet] = inlet[is_inlet]
                result[is_outlet] = outlet[is_outlet]
                return result
            
            @cartesian
            def pressure_dirichlet(self, p: TensorLike, t) -> TensorLike:
                return self.outlet_pressure(p)

            @cartesian
            def inlet_velocity(self, p: TensorLike) -> TensorLike:
                """Compute exact solution of velocity."""
                x = p[..., 0]
                y = p[..., 1]
                result = bm.zeros(p.shape, dtype=bm.float64)
                result[..., 0] =  6 * 1/(0.41)**2 * (0.41 - y) * y
                result[..., 1] = bm.array(0.0)
                return result
            
            @cartesian
            def outlet_pressure(self, p: TensorLike) -> TensorLike:
                """Compute exact solution of pressure."""
                x = p[..., 0]
                y = p[..., 1]
                result = bm.zeros(p.shape[0], dtype=bm.float64)
                return result
            
            @cartesian
            def wall_velocity(self, p: TensorLike, t) -> TensorLike:
                """Compute exact solution of velocity on wall."""
                x = p[..., 0]
                y = p[..., 1]
                result = bm.zeros(p.shape, dtype=bm.float64)
                result[..., 0] = bm.array(0.0)
                result[..., 1] = bm.array(0.0)
                return result
            
            @cartesian
            def wall_pressure(self, p: TensorLike, t) -> TensorLike:
                """Compute exact solution of pressure on wall."""
                x = p[..., 0]
                y = p[..., 1]
                result = bm.zeros(p.shape[0], dtype=p.dtype)
                result[:] = bm.array(0.0)
                return result
            
            @cartesian
            def obstacle_velocity(self, p: TensorLike, t) -> TensorLike:
                """Compute exact solution of velocity on obstacle."""
                x = p[..., 0]
                y = p[..., 1]
                result = bm.zeros(p.shape, dtype=bm.float64)
                result[..., 0] = bm.array(0.0)
                result[..., 1] = bm.array(0.0)
                return result
            
            @cartesian
            def velocity_0(self, p: TensorLike, t) -> TensorLike:
                """Compute exact solution of velocity."""
                x = p[..., 0]
                y = p[..., 1]
                result = bm.zeros(p.shape, dtype=bm.float64)
                return result
            
            @cartesian
            def pressure_0(self, p: TensorLike, t) -> TensorLike:
                x = p[..., 0]
                y = p[..., 1]
                result = bm.zeros(p.shape[0], dtype=p.dtype)
                return result
            
            @cartesian
            def source(self, p: TensorLike, t) -> TensorLike:
                """Compute exact source """
                x = p[..., 0]
                y = p[..., 1]
                result = bm.zeros(p.shape, dtype=bm.float64)
                result[..., 0] = bm.array(0.0)
                result[..., 1] = bm.array(0.0)
                return result
            
            @cartesian
            def is_velocity_boundary(self, p):
                return None
            
            @cartesian
            def is_pressure_boundary(self, p : TensorLike = None) -> TensorLike:
                return 0
            
            @cartesian
            def is_inlet_boundary(self, p: TensorLike) -> TensorLike:
                """Check if point where velocity is defined is on boundary."""
                return bm.abs(p[..., 0]) < self.atol

            @cartesian
            def is_outlet_boundary(self, p: TensorLike) -> TensorLike:
                """Check if point where pressure is defined is on boundary."""
                return bm.abs(p[..., 0] - 2.2) < self.atol
            
            @cartesian
            def is_wall_boundary(self, p: TensorLike) -> TensorLike:
                """Check if point where velocity is defined is on boundary."""
                return (bm.abs(p[..., 1] -0.41) < self.atol) | (bm.abs(p[..., 1] ) < self.atol)
            
            @cartesian
            def is_obstacle_boundary(self, p: TensorLike) -> TensorLike:
                """Check if point where velocity is defined is on boundary."""
                x = p[...,0]
                y = p[...,1]
                cx = cx
                cy = cy
                r = radius
                return (bm.sqrt((x-cx)**2 + (y-cy)**2) - r) < self.atol
        
        options = {
            "mu" : mu,
            "rho" : rho,
            "cx" : cx,
            "cy" : cy,
            "radius" : radius,
            "n_circle" : n_circle,
            "h" : h
        }
        model = PDE(options)
        return (model.mu, model.rho, model.box) + tuple(
            getattr(model, name)
            for name in ["source", "velocity_0", "pressure_0", "velocity_dirichlet", "pressure_dirichlet",
                          "is_velocity_boundary", "is_pressure_boundary", "mesh"]
        )
    

class IncompressibleNSPhysics(CNodeType):
    TITLE: str = "不可压缩 NS 物理变量"
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

        PortConf("inflow", DataType.FLOAT, 0, title="流入速度", default=1.0),
    ]
    OUTPUT_SLOTS = [
        PortConf("source", DataType.FUNCTION, title="源"),
        PortConf("dirichlet_boundary", DataType.FUNCTION, title="边界条件"),
        PortConf("is_boundary", DataType.FUNCTION, title="边界"),
        PortConf("space", DataType.LIST, title="函数空间"),
        PortConf("u0", DataType.TENSOR, title="速度初值"),
        PortConf("p0", DataType.TENSOR, title="压力初值"),
    ]

    @staticmethod
    def run(mesh, utype, u_p, u_gd, ptype, p_p, inflow) -> Union[object]:
        from fealpy.backend import backend_manager as bm
        from fealpy.decorator import cartesian
        from fealpy.functionspace import functionspace

        box = mesh.box

        element_u = (utype.capitalize(), u_p)
        shape_u = (u_gd, -1)
        uspace = functionspace(mesh, element_u, shape=shape_u)

        spaceclass = get_space_class(ptype)
        pspace = spaceclass(mesh, p=p_p)

        eps = 1e-10

        @cartesian
        def is_outflow_boundary(p):
            x = p[...,0]
            y = p[...,1]
            cond1 = bm.abs(x - box[1]) < eps
            cond2 = bm.abs(y-box[2])>eps
            cond3 = bm.abs(y-box[3])>eps
            return (cond1) & (cond2 & cond3) 
        
        @cartesian
        def is_inflow_boundary(p):
            return bm.abs(p[..., 0]-box[0]) < eps
        
        @cartesian
        def is_wall_boundary(p):
            return (bm.abs(p[..., 1] -box[2]) < eps) | \
                (bm.abs(p[..., 1] -box[3]) < eps)
        
        @cartesian
        def is_velocity_boundary(p):
            return ~is_outflow_boundary(p)
            # return None
        
        @cartesian
        def is_pressure_boundary(p=None):
            if p is None:
                return 1
            else:
                return is_outflow_boundary(p) 
                #return bm.zeros_like(p[...,0], dtype=bm.bool)
            # return 0

        @cartesian
        def u_inflow_dirichlet( p):
            x = p[...,0]
            y = p[...,1]
            value = bm.zeros_like(p)
            value[...,0] = inflow
            # value[...,1] = 0
            return value
        
        @cartesian
        def pressure_dirichlet( p, t):
            x = p[...,0]
            y = p[...,1]
            value = bm.zeros_like(x)
            return value

        @cartesian
        def velocity_dirichlet( p, t):
            x = p[...,0]
            y = p[...,1]
            index = is_inflow_boundary(p)
            result = bm.zeros_like(p)
            result[index] = u_inflow_dirichlet(p[index])
            return result
        
        @cartesian
        def source( p, t):
            x = p[..., 0]
            y = p[..., 1]
            result = bm.zeros(p.shape, dtype=bm.float64)
            result[..., 0] = 0
            result[..., 1] = 0
            return result
        
        def dirichlet_boundary():
            u_dirichlet = velocity_dirichlet
            p_dirichlet = pressure_dirichlet
            return (u_dirichlet, p_dirichlet)
        
        def is_boundary():
            is_u_boundary = is_velocity_boundary
            is_p_boundary = is_pressure_boundary
            return (is_u_boundary, is_p_boundary)
        
        def velocity_0(p ,t):
            x = p[...,0]
            y = p[...,1]
            value = bm.zeros(p.shape)
            return value

        def pressure_0( p, t):
            x = p[..., 0]
            val = bm.zeros_like(x)
            return val
            
        space = (uspace, pspace)
        u0 = uspace.interpolate(cartesian(lambda p:velocity_0(p, 0)))
        p0 = pspace.interpolate(cartesian(lambda p:pressure_0(p, 0)))

        return (source, dirichlet_boundary, is_boundary, space, u0, p0)


class IncompressibleNSMathematics(CNodeType):
    TITLE: str = "不可压缩 NS 数学模型"
    PATH: str = "preprocess.modeling"
    DESC: str = """该节点定义了不可压缩 Navier-Stokes 方程的数学模型参数，包括时间项、对流项、压力项和粘性项的系数，以及源项函数。
            这些参数将用于后续的数值求解过程中，确保方程的正确表示和求解。
            
            """
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, title="网格"),
        PortConf("source", DataType.FUNCTION, 1, title="源项"),
    ]
    OUTPUT_SLOTS = [
        PortConf("time_derivative", DataType.FLOAT, title="时间项系数"),
        PortConf("convection", DataType.FLOAT, title="对流项系数"),
        PortConf("pressure", DataType.FLOAT, title="压力项系数"),
        PortConf("viscosity", DataType.FLOAT, title="粘性项系数"),
        PortConf("source", DataType.FUNCTION, title="源项"),
    ]
    def run(mesh, source):
        mu = mesh.nodedata['mu']
        rho = mesh.nodedata['rho']
        time_derivative = rho
        convection = rho
        pressure = 1.0
        viscosity = mu
        return time_derivative, convection, pressure, viscosity, source
    
