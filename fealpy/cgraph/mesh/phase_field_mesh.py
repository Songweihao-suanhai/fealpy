from ..nodetype import CNodeType, PortConf, DataType

__all__ = ['PhaseFieldMesher2d']

class PhaseFieldMesher2d(CNodeType):
    TITLE: str = "相场模型矩形网格建模"
    PATH: str = "examples.CFD"
    INPUT_SLOTS= [
        PortConf("material", DataType.LIST, title="材料属性"),
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

        if material is not None:

            material = material[0]
            for k, value in material.items():
                setattr(mesh, k, value)
                if value is float:
                    mesh.nodedata[k] = material[k] * bm.ones((NN, ), dtype=bm.float64)

        return mesh