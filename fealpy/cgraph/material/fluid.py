from ..nodetype import CNodeType, PortConf, DataType


class IncompressibleNSMaterial(CNodeType):
    TITLE: str = "不可压缩流体材料属性(NS)"
    PATH: str = "preprocess.material"
    INPUT_SLOTS = [
        PortConf("mu", DataType.FLOAT, 0, title="动力粘度", default=0.001),
        PortConf("rho", DataType.FLOAT, 0, title="密度", default=1.0)
    ]
    OUTPUT_SLOTS = [
        PortConf("mu", DataType.FLOAT, title="动力粘度"),
        PortConf("rho", DataType.FLOAT, title="密度")
    ]
    @staticmethod
    def run(mu, rho):
        return mu, rho
        