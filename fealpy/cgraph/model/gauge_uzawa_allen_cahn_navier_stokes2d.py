
from ..nodetype import CNodeType, PortConf, DataType

class ACNSMathmatics(CNodeType):
    TITLE: str = "ACNS 数学模型"
    PATH: str = "examples.CFD"
    INPUT_SLOTS = [
        PortConf("phi", DataType.TENSOR, title="相场"),
        PortConf("u", DataType.TENSOR, title="速度"),
        PortConf("p", DataType.TENSOR, title="压力")
    ]
    OUTPUT_SLOTS = [
        PortConf("equation", DataType.LIST, title="方程"),
        PortConf("boundary", DataType.FUNCTION, title="边界条件"),
        PortConf("is_boundary", DataType.FUNCTION, title="边界"),
        PortConf("x0", DataType.LIST, title="初始值")
    ]
    @staticmethod
    def run(phi, u, p):
        from fealpy.backend import backend_manager as bm
        from fealpy.decorator import barycentric, cartesian
        mesh = phi.space.mesh
        rho = mesh.rho
        mu = mesh.mu
        lam = mesh.lam
        gamma = mesh.gamma
        g = 9.8
        

        
