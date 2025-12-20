from ..nodetype import CNodeType, PortConf, DataType


__all__ = ['RTIMaterial', 'IncompressibleFluid']

class RTIMaterial(CNodeType):
    TITLE: str = "RTI 现象流体材料属性"
    PATH: str = "examples.CFD"
    INPUT_SLOTS = [
        PortConf("rho_up", DataType.FLOAT, 0, title="上层流体密度", default=3.0),
        PortConf("rho_down", DataType.FLOAT, 0, title="下层流体密度", default=1.0),
        PortConf("Re", DataType.FLOAT, 0, title="雷诺数", default=3000.0),
        PortConf("Fr", DataType.FLOAT, 0, title="弗劳德数", default=1.0),
        PortConf("epsilon", DataType.FLOAT, 0, title="界面厚度参数", default=0.01)
    ]
    OUTPUT_SLOTS = [
        PortConf("material", DataType.LIST, title="物理属性")
    ]
    @staticmethod
    def run(rho_up, rho_down, Re, Fr, epsilon):
        from fealpy.backend import backend_manager as bm
        bm.set_backend('pytorch')
        bm.set_default_device('cpu')
        def rho(phi):
            result = phi.space.function()
            result[:] = (rho_up - rho_down)/2 * phi[:]
            result[:] += (rho_up + rho_down)/2 
            return result
        Pe = 1/epsilon

        material = [{
            'rho': rho, 
            'Re': Re, 
            'Fr': Fr, 
            'epsilon': epsilon,
            'Pe': Pe
            }]

        return material
    
class IncompressibleFluid(CNodeType):
    TITLE: str = "不可压缩流体材料属性"
    PATH: str = "examples.CFD"
    INPUT_SLOTS = [
        PortConf("material", DataType.MENU, 0, title="材料", default="water", items=["water", "air", "custom"]),
        PortConf("mu", DataType.FLOAT, 0, title="动力粘度", default=0.001),
        PortConf("rho", DataType.FLOAT, 0, title="密度", default=1.0)
    ]
    OUTPUT_SLOTS = [
        PortConf("mp", DataType.LIST, title="材料属性")
    ]
    @staticmethod
    def run(material, mu, rho):
        if material == "water":
            mp = [{
                'mu': 0.001, 
                'rho': 1000.0
                }]
        elif material == "air": 
            mp = [{
                'mu': 0.0000181, 
                'rho': 1.225
                }] 
        else:
            mp = [{
                'mu': mu, 
                'rho': rho
                }]
        return mp



