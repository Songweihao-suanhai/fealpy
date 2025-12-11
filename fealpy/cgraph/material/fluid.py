from ..nodetype import CNodeType, PortConf, DataType


__all__ = ['RTIMaterial']

class RTIMaterial(CNodeType):
    TITLE: str = "RTI 现象流体物理属性"
    PATH: str = "preprocess.material"
    INPUT_SLOTS = [
        PortConf("rho_up", DataType.FLOAT, 0, title="上层流体密度", default=3.0),
        PortConf("rho_down", DataType.FLOAT, 0, title="下层流体密度", default=1.0),
        PortConf("Re", DataType.FLOAT, 0, title="雷诺数", default=3000.0),
        PortConf("Fr", DataType.FLOAT, 0, title="弗劳德数", default=1.0),
        PortConf("epsilon", DataType.FLOAT, 0, title="界面厚度参数", default=0.01)
    ]
    OUTPUT_SLOTS = [
        PortConf("rho", DataType.FUNCTION, title="密度函数"),
        PortConf("Re", DataType.FLOAT, title="雷诺数"),
        PortConf("Fr", DataType.FLOAT, title="弗劳德数"),
        PortConf("epsilon", DataType.FLOAT, title="界面厚度参数"),
        PortConf("Pe", DataType.FLOAT, title="Peclet 数")
    ]
    @staticmethod
    def run(rho_up, rho_down, Re, Fr, epsilon):
        def rho(phi):
            result = phi.space.function()
            result[:] = (rho_up - rho_down)/2 * phi[:]
            result[:] += (rho_up + rho_down)/2 
            return result
        
        Pe = 1/epsilon
        return rho, Re, Fr, epsilon, Pe


