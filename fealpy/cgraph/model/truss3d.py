from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["Truss3dData", "TrussTower3d"]


class Truss3dData(CNodeType):
    r"""A 25-bar truss structure model.

    This node generates the node coordinates and cell connectivity for a 
    classic 25-bar space truss.

    Outputs:
        node (tensor): The node coordinates of the truss.
        cell (tensor): The cell connectivity of the truss (edges).
    """
    TITLE: str = "25杆桁架几何"
    PATH: str = "模型.几何"
    DESC: str = "生成一个经典的25杆空间桁架的节点坐标和单元连接关系。"

    INPUT_SLOTS = []

    OUTPUT_SLOTS = [
        PortConf("node", DataType.TENSOR, desc="桁架的节点坐标", title="节点坐标"),
        PortConf("cell", DataType.TENSOR, desc="桁架的单元连接关系", title="单元"),
    ]

    @staticmethod
    def run():
        from fealpy.backend import backend_manager as bm

        node = bm.array([
            [-950, 0, 5080], [950, 0, 5080], [-950, 950, 2540],
            [950, 950, 2540], [950, -950, 2540], [-950, -950, 2540],
            [-2540, 2540, 0], [2540, 2540, 0], [2540, -2540, 0],
            [-2540, -2540, 0]], dtype=bm.float64)
        edge = bm.array([
            [0, 1], [3, 0], [1, 2], [1, 5], [0, 4],
            [1, 3], [1, 4], [0, 2], [0, 5], [2, 5],
            [4, 3], [2, 3], [4, 5], [2, 9], [6, 5],
            [8, 3], [7, 4], [6, 3], [2, 7], [9, 4],
            [8, 5], [9, 5], [2, 6], [7, 3], [8, 4]], dtype=bm.int32)
        
        cell = edge

        return node, cell


class TrussTower3d(CNodeType):
    r"""3D Truss Tower Model.
    
     Inputs:
        dov (float): Outer diameter of vertical rods (m).
        div (float): Inner diameter of vertical rods (m).
        doo (float): Outer diameter of other rods (m).
        dio (float): Inner diameter of other rods (m).
        load_total (float): Total vertical load applied at top nodes (N).
        
    Outputs:
        Av (float): Cross-sectional area of vertical rods (m²).
        Ao (float): Cross-sectional area of other rods (m²).
        Iv (float): Area moment of inertia of vertical rods (m⁴).
        Io (float): Area moment of inertia of other rods (m⁴).
        I1 (float): Structural inertia in depth direction (m⁴).
        I2 (float): Structural inertia in width direction (m⁴).
        external_load (Function): Global load vector.
        dirichlet_dof (Function): Dirichlet boundary DOF indices.
    """
    TITLE: str = "桁架塔模型"
    PATH: str = "preprocess.modeling"
    DESC: str = "生成三维桁架塔结构模型，包含管状截面几何参数、结构惯性矩、外部载荷向量及边界条件，用于桁架结构力学分析。"
    INPUT_SLOTS = [
        PortConf("dov", DataType.FLOAT, 1,  desc="竖向杆件的外径", title="竖杆外径", default=0.015),
        PortConf("div", DataType.FLOAT, 1,  desc="竖向杆件的内径", title="竖杆内径", default=0.010),
        PortConf("doo", DataType.FLOAT, 1,  desc="其他杆件的外径", title="其他杆外径", default=0.010),
        PortConf("dio", DataType.FLOAT, 1,  desc="其他杆件的内径", title="其他杆内径", default=0.007),
        PortConf("load_total", DataType.FLOAT, 1,  desc="施加在塔顶节点的总竖向载荷", title="总载荷", default=84820.0)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("Av", DataType.FLOAT,  desc="竖向杆件的截面积", title="竖杆截面积"),
        PortConf("Ao", DataType.FLOAT,  desc="其他杆件的截面积", title="其他杆截面积"),
        PortConf("Iv", DataType.FLOAT,  desc="竖向杆件的面积惯性矩", title="竖杆惯性矩"),
        PortConf("Io", DataType.FLOAT,  desc="其他杆件的面积惯性矩", title="其他杆惯性矩"),
        PortConf("I1", DataType.FLOAT,  desc="深度方向的结构惯性矩", title="结构惯性矩I1"),
        PortConf("I2", DataType.FLOAT,  desc="宽度方向的结构惯性矩", title="结构惯性矩I2"),
        PortConf("external_load", DataType.FUNCTION, desc="全局载荷向量，表示总载荷如何分布到顶部节点", title="外部载荷"),
        PortConf("dirichlet_dof", DataType.FUNCTION, desc="Dirichlet边界条件的自由度索引", title="边界自由度")
    ]


    @staticmethod
    def run(**options):
        from fealpy.csm.model.truss.truss_tower_data_3d import TrussTowerData3D
        
        model = TrussTowerData3D(
            dov=options.get("dov"),
            div=options.get("div"),
            doo=options.get("doo"),
            dio=options.get("dio")
        )
        
        load_total = options.get("load_total")
        
        return tuple(
            getattr(model, name)
            for name in ["Av", "Ao", "Iv", "Io", "I1", "I2", "external_load", "dirichlet_dof"]
        )