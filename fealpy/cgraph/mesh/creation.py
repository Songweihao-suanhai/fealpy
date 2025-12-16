
from ..nodetype import CNodeType, PortConf, DataType
from .utils import get_mesh_class

__all__ = ["CreateMesh", "PolygonMesh2d",
           "DLDMicrofluidicChipMesh2d", "DLDMicrofluidicChipMesh3d",
           "NACA4Mesh2d", "RTIMesher2d"]


class CreateMesh(CNodeType):
    r"""Create a mesh object.This node generates a mesh of the specified type 
    using given node and cell data.

    Inputs:
        mesh_type (str): Type of mesh to granerate.
        Supported values: "triangle", "quadrangle", "tetrahedron", "hexahedron".Default is "edgemesh".
        node(tensor):Coordinates of mesh nodes.
        cell(tensor):Connectivity of mesh cells.

    Outputs:
        mesh (MeshType): The mesh object created.
    """
    TITLE: str = "构造网格"
    PATH: str = "preprocess.mesher"
    DESC: str = """从网格点坐标(node)和单元数据(cell)直接生成网格对象。
                该节点直接引用网格点坐标和单元数据张量，并将其解释为网格。
                使用例子：通过两个“数据.张量”节点分别创建网格点坐标张量和单元数据张量，连接到该节点的相应输入上，
                再将该节点连接到输出，即可查看网格构造效果。
                """
    INPUT_SLOTS = [
        PortConf("mesh_type", DataType.MENU, 0, title="网格类型", default="edge", 
                 items=["triangle", "quadrangle", "tetrahedron", "hexahedron", "edge"]),
        PortConf("node", DataType.TENSOR, 1, title="节点坐标"),
        PortConf("cell", DataType.TENSOR, 1, title="单元")
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]

    @staticmethod
    def run(mesh_type, node, cell):
        MeshClass = get_mesh_class(mesh_type)
        kwds = {"node": node, "cell": cell}
        return MeshClass(**kwds)


class PolygonMesh2d(CNodeType):
    r"""Create a polygon mesh object.
    
    Inputs:
        mesh_type (menu): Type of mesh to granerate.
        vertices(list): The vertices of the polygon.
        h(float): The height of the polygon.

    Outputs:
        mesh (PolygonMesh): The polygon mesh object created.
    """
    TITLE: str = "多边形网格"
    PATH: str = "preprocess.mesher"
    DESC: str = """该节点生成多边形网格, 依据输入的多边形顶点坐标自动构建多边形网格."""
    
    INPUT_SLOTS = [
        PortConf("mesh_type", DataType.MENU, 0, title="网格类型", default="triangle", items=["triangle", "quadrangle"]),
        PortConf("vertices", DataType.LIST, 1, title="多边形顶点"),
        PortConf("h", DataType.FLOAT, 1, default=0.02, title="网格尺寸")
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")       
    ]
    
    @staticmethod
    def run(mesh_type, vertices, h):
        MeshClass = get_mesh_class(mesh_type)
        kwds = {"vertices": vertices, "h": h}
        return  MeshClass.from_polygon_gmsh(**kwds)
    

class DLDMicrofluidicChipMesh2d(CNodeType):
    r"""Create a mesh in a DLD microfluidic chip-shaped 2D area.

    Inputs:
        init_point X (float, optional): Initial point of the chip.
        init_point Y (float, optional): Initial point of the chip.
        chip_height (float, optional): Height of the chip.
        inlet_length (float, optional): Length of the inlet.
        outlet_length (float, optional): Length of the outlet.
        radius (float, optional): Radius of the micropillars.
        n_rows (int, optional): Number of rows of micropillars.
        n_cols (int, optional): Number of columns of micropillars.
        tan_angle (float, optional): Tangent value of the angle of deflection.
        n_stages (int, optional): Number of periods of micropillar arrays.
        stage_length (float, optional): Length of a single period.
        lc (float, optional): Target mesh size.

    Outputs:
        mesh (Mesh): The mesh object created.
        radius (float): Radius of the micropillars.
        centers (tensor): Coordinates of the centers of the micropillars.
        inlet_boundary (tensor): Inlet boundary.
        outlet_boundary (tensor): Outlet boundary.
        wall_boundary (tensor): Wall boundary of the channel.
    """
    TITLE: str = "二维 DLD 微流芯片网格"
    PATH: str = "preprocess.mesher"
    DESC: str = """该节点生成二维DLD微流控芯片的网格剖分, 依据几何与周期参数自动构建微柱
                阵列及流道边界，为微流控芯片数值模拟提供几何与网格基础。"""
    INPUT_SLOTS = [
        PortConf("init_point_x", DataType.FLOAT, 1, default=0.0, title="初始点 X"),
        PortConf("init_point_y", DataType.FLOAT, 1, default=0.0, title="初始点 Y"),
        PortConf("chip_height", DataType.FLOAT, 1, default=1.0, title="芯片长度"),
        PortConf("inlet_length", DataType.FLOAT, 1, default=0.1, title="入口宽度"),
        PortConf("outlet_length", DataType.FLOAT, 1, default=0.1, title="出口宽度"),
        PortConf("radius", DataType.FLOAT, 1, default=1 / (3 * 4 * 3), title="微柱半径"),
        PortConf("n_rows", DataType.INT, 1, default=8, title="行数"),
        PortConf("n_cols", DataType.INT, 1, default=4, title="列数"),
        PortConf("tan_angle", DataType.FLOAT, 1, default=1/7, title="偏转角正切值"),
        PortConf("n_stages", DataType.INT, 1, default=3, title="微柱阵列周期数"),
        PortConf("stage_length", DataType.FLOAT, 1, default=1.4, title="单周期长度"),
        PortConf("lc", DataType.FLOAT, 1, default=0.02, title="网格尺寸")
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格"),
        PortConf("radius", DataType.FLOAT, title="微柱半径"),
        PortConf("centers", DataType.TENSOR, title="微柱圆心坐标"),
        PortConf("inlet_boundary", DataType.TENSOR, title="入口边界"),
        PortConf("outlet_boundary", DataType.TENSOR, title="出口边界"),
        PortConf("wall_boundary", DataType.TENSOR, title="通道壁面边界")
    ]

    @staticmethod
    def run(**options):
        from fealpy.geometry import DLDMicrofluidicChipModeler
        from fealpy.mesher import DLDMicrofluidicChipMesher
        import gmsh

        options = {
            "init_point" : (options.get("init_point_x"), options.get("init_point_y")),
            "chip_height" : options.get("chip_height"),
            "inlet_length" : options.get("inlet_length"),
            "outlet_length" : options.get("outlet_length"),
            "radius" : options.get("radius"),
            "n_rows" : options.get("n_rows"),
            "n_cols" : options.get("n_cols"),
            "tan_angle" : options.get("tan_angle"),
            "n_stages" : options.get("n_stages"),
            "stage_length" : options.get("stage_length"),
            "lc" : options.get("lc")
        }

        gmsh.initialize()
        modeler = DLDMicrofluidicChipModeler(options)
        modeler.build(gmsh)
        mesher = DLDMicrofluidicChipMesher(options)
        mesher.generate(modeler, gmsh)
        gmsh.finalize()

        return (mesher.mesh, mesher.radius, mesher.centers, mesher.inlet_boundary, 
                mesher.outlet_boundary, mesher.wall_boundary)


class DLDMicrofluidicChipMesh3d(CNodeType):
    r"""Generate a 3D mesh for a DLD (Deterministic Lateral Displacement) microfluidic chip.

    Inputs:
        init_point_x (float): X-coordinate of the initial reference point.
        init_point_y (float): Y-coordinate of the initial reference point.
        chip_height (float): Total height (length) of the chip domain.
        inlet_length (float): Inlet channel width.
        outlet_length (float): Outlet channel width.
        thickness (float): Chip thickness (z-direction dimension).
        radius (float): Radius of each micropillar.
        n_rows (int): Number of micropillar rows in the array.
        n_cols (int): Number of micropillar columns in the array.
        tan_angle (float): Tangent of the DLD array inclination angle (defines lateral shift).
        n_stages (int): Number of periodic stages (DLD array periods).
        stage_length (float): Length of one periodic stage in the array.
        lc (float): Characteristic mesh size (element size).

    Outputs:
        mesh (Mesh): The generated 3D mesh of the microfluidic chip.
        thickness (float): The effective chip thickness used for meshing.
        radius (float): The micropillar radius used in the geometry.
        centers (Tensor): Coordinates of the micropillar centers.
        inlet_boundary (Tensor): Node or face data defining the inlet boundary.
        outlet_boundary (Tensor): Node or face data defining the outlet boundary.
        wall_boundary (Tensor): Node or face data defining the channel wall boundaries.
    """
    TITLE: str = "三维 DLD 微流芯片网格"
    PATH: str = "preprocess.mesher"
    DESC: str = """该节点生成三维DLD微流控芯片的网格剖分, 依据几何与周期参数自动构建微柱
                阵列及流道边界，为微流控芯片数值模拟提供几何与网格基础。"""
    INPUT_SLOTS = [
        PortConf("init_point_x", DataType.FLOAT, 1, default=0.0, title="初始点 X"),
        PortConf("init_point_y", DataType.FLOAT, 1, default=0.0, title="初始点 Y"),
        PortConf("chip_height", DataType.FLOAT, 1, default=1.0, title="芯片长度"),
        PortConf("inlet_length", DataType.FLOAT, 1, default=0.2, title="入口宽度"),
        PortConf("outlet_length", DataType.FLOAT, 1, default=0.2, title="出口宽度"),
        PortConf("thickness", DataType.FLOAT, 1, default=0.1, title="芯片厚度"),
        PortConf("radius", DataType.FLOAT, 1, default=1 / (3 * 5), title="微柱半径"),
        PortConf("n_rows", DataType.INT, 1, default=3, title="行数"),
        PortConf("n_cols", DataType.INT, 1, default=3, title="列数"),
        PortConf("tan_angle", DataType.FLOAT, 1, default=1/7, title="偏转角正切值"),
        PortConf("n_stages", DataType.INT, 1, default=2, title="微柱阵列周期数"),
        PortConf("stage_length", DataType.FLOAT, 1, default=1.4, title="单周期长度"),
        PortConf("lc", DataType.FLOAT, 1, default=0.02, title="网格尺寸")
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格"),
        PortConf("thickness", DataType.FLOAT, title="芯片厚度"),
        PortConf("radius", DataType.FLOAT, title="微柱半径"),
        PortConf("centers", DataType.TENSOR, title="微柱圆心坐标"),
        PortConf("inlet_boundary", DataType.TENSOR, title="入口边界"),
        PortConf("outlet_boundary", DataType.TENSOR, title="出口边界"),
        PortConf("wall_boundary", DataType.TENSOR, title="通道壁面边界")
    ]

    @staticmethod
    def run(**options):
        from fealpy.geometry import DLDMicrofluidicChipModeler3D
        from fealpy.mesher import DLDMicrofluidicChipMesher3D
        import gmsh

        options = {
            "init_point" : (options.get("init_point_x"), options.get("init_point_y")),
            "chip_height" : options.get("chip_height"),
            "inlet_length" : options.get("inlet_length"),
            "outlet_length" : options.get("outlet_length"),
            "thickness": options.get("thickness"),
            "radius" : options.get("radius"),
            "n_rows" : options.get("n_rows"),
            "n_cols" : options.get("n_cols"),
            "tan_angle" : options.get("tan_angle"),
            "n_stages" : options.get("n_stages"),
            "stage_length" : options.get("stage_length"),
            "lc" : options.get("lc")
        }

        gmsh.initialize()
        modeler = DLDMicrofluidicChipModeler3D(options)
        modeler._apply_auto_config()
        modeler.build(gmsh)
        mesher = DLDMicrofluidicChipMesher3D(options)
        mesher.generate(modeler, gmsh)
        gmsh.finalize()

        return (mesher.mesh, mesher.options.get('thickness'),mesher.radius, mesher.centers, mesher.inlet_boundary, 
                mesher.outlet_boundary, mesher.wall_boundary)


class NACA4Mesh2d(CNodeType):
    TITLE: str = "二维 NACA 四位数翼型几何建模与网格生成"
    PATH: str = "examples.CFD"
    DESC: str = """该节点生成二维 NACA4 系列翼型的网格剖分, 依据翼型参数自动构建翼型几何形状及
                流道边界，为翼型流场数值模拟提供几何与网格基础。"""
    INPUT_SLOTS = [
        PortConf("m", DataType.FLOAT, 0, default=0.0, title="最大弯度"),
        PortConf("p", DataType.FLOAT, 0, default=0.0, title="最大弯度位置"),
        PortConf("t", DataType.FLOAT, 0, default=0.12, title="相对厚度"),
        PortConf("c", DataType.FLOAT, 0, default=1.0, title="弦长"),
        PortConf("alpha", DataType.FLOAT, 0, default=0.0, title="攻角"),
        PortConf("N", DataType.INT, 0, default=200, title="翼型轮廓分段数"),
        PortConf("box", DataType.TEXT, 0, default=[-0.5, 2.7, -0.4, 0.4], title="求解域"),
        PortConf("h", DataType.FLOAT, 0, default=0.02, title="全局网格尺寸"),
        PortConf("thickness", DataType.FLOAT, 0, default=None, title="边界层厚度"),
        PortConf("ratio", DataType.FLOAT, 0, default=2.4, title="边界层增长率"),
        PortConf("size", DataType.FLOAT, 0, default=None, title="翼型附近网格尺寸"),
        PortConf("material", DataType.NONE, 1, title="材料"),
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]
    def run(**options):
        import math
        from fealpy.backend import backend_manager as bm
        from fealpy.mesher.naca4_mesher import NACA4Mesher
        from fealpy.decorator import cartesian

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
        h = options.get("h", 0.02)
        thickness = options.get("thickness", h/10)
        ratio = options.get("ratio", 2.4)
        size = options.get("size", h/50)
        
        # singular_points = bm.array([[0, 0], [0.97476, 0.260567]], dtype=bm.float64)
        singular_points = bm.array([[0, 0], [1.00, 0.0]], dtype=bm.float64)
        hs = [h/3, h/3] 
        mesher = NACA4Mesher(m , p , t, c, alpha, N, box, singular_points)
        mesh = mesher.init_mesh(h, hs, is_quad=0, thickness = thickness, ratio=ratio, size=size)
        mesh.box = box
        NN = mesh.number_of_nodes()
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
        
        mesh.is_inlet_boundary = is_inlet_boundary
        mesh.is_outlet_boundary = is_outlet_boundary
        mesh.is_wall_boundary = is_wall_boundary

        for k, value in material.items():
            setattr(mesh, k, value)
            mesh.nodedata[k] = material[k] * bm.ones((NN, ), dtype=bm.float64)

        return mesh
    
class FlowPastCylinder2d(CNodeType):
    TITLE: str = "二维圆柱绕流几何建模与网格生成"
    PATH: str = "preprocess.mesher"
    INPUT_SLOTS= [
        PortConf("box", DataType.TEXT, 0, default=(0.0, 2.2, 0.0, 0.41), title="求解域"),
        PortConf("center", DataType.TEXT, 0, default=(0.2, 0.2), title="圆心坐标"),
        PortConf("radius", DataType.FLOAT, 0, default=0.05, title="圆柱半径"),
        PortConf("n_circle", DataType.INT, 0, default=100, title="圆柱周围点数"),
        PortConf("h", DataType.FLOAT, 0, default=0.01, title="全局网格尺寸")
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]
    @staticmethod
    def run(box, center, radius, n_circle, h):
        import gmsh 
        import math
        from fealpy.backend import backend_manager as bm
        from fealpy.mesh import TriangleMesh 
        box = bm.tensor(eval(box, None, vars(math)), dtype=bm.float64)
        center = bm.tensor(eval(center, None, vars(math)), dtype=bm.float64)
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
        mesh = TriangleMesh(node_coords, tri_nodes)
        mesh.box = box
        mesh.center = center
        return mesh


class RTIMesher2d(CNodeType):
    TITLE: str = "二维 RTI 问题几何建模与网格生成"
    PATH: str = "examples.CFD"
    INPUT_SLOTS= [
        PortConf("material", DataType.LIST, title="物理属性"),
        PortConf("box", DataType.TEXT, 0, title="求解域"),
        PortConf("nx", DataType.INT, 0, default=64, title="x方向单元数"),
        PortConf("ny", DataType.INT, 0, default=256, title="y方向单元数"),
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]
    @staticmethod
    def run(material, box, nx, ny):
        from fealpy.backend import backend_manager as bm
        from fealpy.mesh import TriangleMesh
        from fealpy.decorator import cartesian
        import math
        box = bm.tensor(eval(box, None, vars(math)), dtype=bm.float64)
        mesh = TriangleMesh.from_box(box = box, nx = nx, ny = ny)
        mesh.box = box
        material = material[0]
        NN = mesh.number_of_nodes()
        eps = 1e-10

        @cartesian
        def is_up_boundary(p):
            tag_up = bm.abs(p[..., 1] - box[3]) < eps
            return tag_up
        
        @cartesian
        def is_down_boundary(p):
            tag_down = bm.abs(p[..., 1] - box[2]) < eps
            return tag_down
        
        @cartesian
        def is_left_boundary(p):
            tag_left = bm.abs(p[..., 0] - box[0]) < eps
            return tag_left
        
        @cartesian
        def is_right_boundary(p):
            tag_right = bm.abs(p[..., 0] - box[1]) < eps
            return tag_right
        
        mesh.is_up_boundary = is_up_boundary
        mesh.is_down_boundary = is_down_boundary
        mesh.is_left_boundary = is_left_boundary
        mesh.is_right_boundary = is_right_boundary

        for k, value in material.items():
            setattr(mesh, k, value)
            mesh.nodedata[k] = material[k] * bm.ones((NN, ), dtype=bm.float64)

        return mesh



