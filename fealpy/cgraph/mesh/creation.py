
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
    
