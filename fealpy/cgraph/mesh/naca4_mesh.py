from fealpy.cgraph.nodetype import CNodeType, PortConf, DataType

__all__ = ['NACA4Geometry2d', 'NACA4Mesh2d']

class NACA4Geometry2d(CNodeType):
    TITLE: str = "二维 NACA 四位数翼型几何建模"
    PATH: str = "examples.CFD"
    DESC: str = """该节点生成二维 NACA4 系列翼型的几何数据，依据翼型参数自动构建翼型几何形状，
                为翼型流场数值模拟提供几何基础。"""
    INPUT_SLOTS = [
        PortConf("m", DataType.FLOAT, 0, default=0.0, title="最大弯度"),
        PortConf("p", DataType.FLOAT, 0, default=0.0, title="最大弯度位置"),
        PortConf("t", DataType.FLOAT, 0, default=0.12, title="相对厚度"),
        PortConf("c", DataType.FLOAT, 0, default=1.0, title="弦长"),
        PortConf("alpha", DataType.FLOAT, 0, default=0.0, title="攻角"),
        PortConf("N", DataType.INT, 0, default=200, title="翼型采样点数"),
        PortConf("box", DataType.TEXT, 0, default=[-0.5, 2.7, -0.4, 0.4], title="求解域"),
        PortConf("material", DataType.NONE, 1, title="材料"),
    ]
    OUTPUT_SLOTS = [
        PortConf("geometry", DataType.LIST, title="几何数据")
    ]
    
    @staticmethod
    def run(**options):
        import math
        from fealpy.backend import backend_manager as bm
        from fealpy.mesher.naca4_mesher import NACA4Mesher

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

        mesher = NACA4Mesher(m , p , t, c, alpha, N, box, singular_points)

        eps = 1e-10
        def is_inlet_boundary(p):
            x = p[..., 0]
            return bm.abs(x - box[0]) < eps
        def is_outlet_boundary(p):
            x = p[...,0]
            y = p[...,1]
            cond1 = bm.abs(x - box[1]) < eps
            cond2 = bm.abs(y-box[2])>eps
            cond3 = bm.abs(y-box[3])>eps
            return (cond1) & (cond2 & cond3) 
        def is_wall_boundary(p):
            y = p[..., 1]
            return (bm.abs(y - box[2]) < eps) | (bm.abs(y - box[3]) < eps)
        
        mesher.is_inlet_boundary = is_inlet_boundary
        mesher.is_outlet_boundary = is_outlet_boundary
        mesher.is_wall_boundary = is_wall_boundary
        mesher.material = material

        geometry = [
            {"mesher": mesher}
        ]

        return geometry

class NACA4Mesh2d(CNodeType):
    TITLE: str = "二维 NACA 四位数翼型几何建模与网格生成"
    PATH: str = "examples.CFD"
    DESC: str = """该节点生成二维 NACA4 系列翼型的网格剖分, 依据翼型参数自动构建翼型几何形状及
                流道边界，为翼型流场数值模拟提供几何与网格基础。"""
    INPUT_SLOTS = [
        PortConf("geometry", DataType.LIST, 1, title="几何数据"),
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
        mesher = geometry[0]['mesher']
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