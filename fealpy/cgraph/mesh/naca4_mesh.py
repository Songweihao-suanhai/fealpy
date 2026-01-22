from fealpy.cgraph.nodetype import CNodeType, PortConf, DataType

__all__ = ['NACA4Geometry2d', 'NACA4Mesh2d']

class NACA4Geometry2d(CNodeType):
    TITLE: str = "二维 NACA 四位数翼型几何建模"
    PATH: str = "examples.CFD"
    DESC: str = """该节点生成二维 NACA4 系列翼型的几何数据，依据翼型参数自动构建翼型几何形状，
                为翼型流场数值模拟提供几何基础。"""
    INPUT_SLOTS = [
        PortConf("material", DataType.NONE, 1, title="材料"),
        PortConf("box", DataType.TEXT, 0, default=[-0.5, 2.7, -0.4, 0.4], title="求解域"),
        PortConf("m", DataType.FLOAT, 0, default=0.0, title="最大弯度"),
        PortConf("p", DataType.FLOAT, 0, default=0.0, title="最大弯度位置"),
        PortConf("t", DataType.FLOAT, 0, default=0.12, title="相对厚度"),
        PortConf("c", DataType.FLOAT, 0, default=1.0, title="弦长"),
        PortConf("alpha", DataType.FLOAT, 0, default=0.0, title="攻角"),
        PortConf("N", DataType.INT, 0, default=200, title="翼型采样点数"),
        PortConf("is_velocity_boundary", DataType.TEXT, 0, default="['inlet', 'wall', 'airfoil']", title="速度边界"),
        PortConf("is_pressure_boundary", DataType.TEXT, 0, default="['outlet']", title="压力边界")
    ]
    OUTPUT_SLOTS = [
        PortConf("geometry", DataType.LIST, title="几何信息")
    ]
    
    @staticmethod
    def run(**options):
        import math
        import ast
        from fealpy.backend import backend_manager as bm
        from fealpy.mesher.naca4_mesher import NACA4Mesher
        from fealpy.decorator import variantmethod, cartesian

        m = options.get("m", 0.0)
        p = options.get("p", 0.0)
        t = options.get("t", 0.12)
        c = options.get("c", 1.0)
        alpha = options.get("alpha", 0.0)
        N = options.get("N", 200)
        box = options.get("box")
        box = bm.tensor(eval(box, None, vars(math)), dtype=bm.float64)
        material = options.get("material", None)
        material = material[0]

        theta = alpha / 180.0 * bm.pi
        singular_points = bm.array([[0.0, 0.0], 
                                    [c * bm.cos(theta), c * bm.sin(theta)]], 
                                   dtype=bm.float64)

        model = NACA4Mesher(m , p , t, c, alpha, N, box, singular_points)
        eps = 1e-10

        @variantmethod("inlet")
        def is_boundary(p):
            x = p[..., 0]
            return bm.abs(x - box[0]) < eps
        
        @is_boundary.register("outlet")
        def is_boundary(p):
            x = p[...,0]
            y = p[...,1]
            cond1 = bm.abs(x - box[1]) < eps
            cond2 = bm.abs(y-box[2])>eps
            cond3 = bm.abs(y-box[3])>eps
            return (cond1) & (cond2 & cond3) 
        
        @is_boundary.register("wall")
        def is_boundary(p):
            y = p[..., 1]
            return (bm.abs(y - box[2]) < eps) | (bm.abs(y - box[3]) < eps)
        
        @is_boundary.register("airfoil")
        def is_boundary(p):
            x = p[..., 0]
            y = p[..., 1]
            is_inlet_boundary = is_boundary["inlet"]
            is_outlet_boundary = is_boundary["outlet"]
            is_wall_boundary = is_boundary["wall"]
            cond = ~(is_inlet_boundary(p) | is_outlet_boundary(p) | is_wall_boundary(p))
            return cond
        
        @cartesian
        def is_u_boundary(p):
            is_u_bd = options.get("is_velocity_boundary")
            is_u_bd = ast.literal_eval(is_u_bd)

            for bd in is_u_bd:
                bd_func = is_boundary[bd]
                if bd == is_u_bd[0]:
                    value = bd_func(p)
                else:
                    value = value | bd_func(p)
            return value
        
        @cartesian
        def is_p_boundary(p = None):
            if p is None:
                return 1
            is_p_bd = options.get("is_pressure_boundary")
            is_p_bd = ast.literal_eval(is_p_bd)
            for bd in is_p_bd:
                bd_func = is_boundary[bd]
                if bd == is_p_bd[0]:
                    value = bd_func(p)
                else:
                    value = value | bd_func(p)
            return value
        
        model.is_boundary = is_boundary
        model.is_velocity_boundary = is_u_boundary
        model.is_pressure_boundary = is_p_boundary
        model.material = material
        geometry = {"model": model}
    
        return geometry

class NACA4Mesh2d(CNodeType):
    TITLE: str = "二维 NACA 四位数翼型网格生成"
    PATH: str = "examples.CFD"
    DESC: str = """该节点生成二维 NACA4 系列翼型的网格剖分, 依据翼型参数自动构建翼型几何形状及
                流道边界，为翼型流场数值模拟提供几何与网格基础。"""
    INPUT_SLOTS = [
        PortConf("geometry", DataType.LIST, 1, title="几何信息"),
        PortConf("h", DataType.FLOAT, 0, default=0.02, title="全局网格尺寸"),
        PortConf("thickness", DataType.FLOAT, 0, default=None, title="边界层厚度"),
        PortConf("ratio", DataType.FLOAT, 0, default=2.4, title="边界层增长率"),
        PortConf("le_size", DataType.FLOAT, 0, default=None, title="前缘附近网格尺寸"),
        PortConf("te_size", DataType.FLOAT, 0, default=None, title="后缘附近网格尺寸"),
        PortConf("size", DataType.FLOAT, 0, default=None, title="翼型附近网格尺寸"),
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]
    def run(**options):
        from fealpy.backend import backend_manager as bm

        geometry = options.get("geometry", None)
        mesher = geometry.get("model")
        material = mesher.material
        h = options.get("h", 0.02)
        thickness = options.get("thickness", h/10)
        ratio = options.get("ratio", 2.4)
        h_le = options.get("le_size", h/3)
        h_te = options.get("te_size", h/3)
        size = options.get("size", h/50)

        hs = [h_le, h_te] 
        mesh = mesher.init_mesh(h, hs, 
                                is_quad=0, 
                                thickness = thickness, 
                                ratio=ratio, 
                                size=size)
        mesh.geo = mesher
        mesh.box = mesher.box
        NN = mesh.number_of_nodes()
        
        for k, value in material.items():
            setattr(mesh, k, value)
            mesh.nodedata[k] = material[k] * bm.ones((NN, ), dtype=bm.float64)

        return mesh