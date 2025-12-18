
from ..nodetype import CNodeType, PortConf, DataType
from .utils import get_mesh_class

__all__ = ["CreateMesh", "PolygonMesh2d"]


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

