from ..nodetype import CNodeType, PortConf, DataType


__all__ = ['MultiphaseFlowMaterial', 'IncompressibleFluid']

class MultiphaseFlowMaterial(CNodeType):
    TITLE: str = "两相流体材料属性"
    PATH: str = "examples.CFD"
    INPUT_SLOTS = [
        PortConf("rho0", DataType.FLOAT, 0, title="第一液相密度", default=3.0),
        PortConf("rho1", DataType.FLOAT, 0, title="第二液相密度", default=1.0),
        PortConf("mu0", DataType.FLOAT, 0, title="第一液相粘度", default=0.0011),
        PortConf("mu1", DataType.FLOAT, 0, title="第二液相粘度", default=0.0011),
        PortConf("lam", DataType.FLOAT, 0, title="应力系数", default=0.001),
        PortConf("gamma", DataType.FLOAT, 0, title="迁移率参数", default=0.02),
        PortConf("Re", DataType.FLOAT, 0, title="雷诺数", default=3000.0),
        PortConf("Fr", DataType.FLOAT, 0, title="弗劳德数", default=1.0),
        PortConf("epsilon", DataType.FLOAT, 0, title="界面厚度参数", default=0.01)
    ]
    OUTPUT_SLOTS = [
        PortConf("material", DataType.DICT, title="物理属性")
    ]
    @staticmethod
    def run(rho0, rho1, mu0, mu1, lam, gamma, Re, Fr, epsilon):
        from fealpy.backend import backend_manager as bm
        # bm.set_backend('pytorch')
        # bm.set_default_device('cpu')
        def rho(phi):
            result = phi.space.function()
            result[:] = (rho0 - rho1)/2 * phi[:]
            result[:] += (rho0 + rho1)/2 
            return result

        Pe = 1/epsilon

        material = {
            'rho0': rho0,
            'rho1': rho1,
            'rho': rho, 
            'mu0': mu0,
            'mu1': mu1,
            'lam': lam, 
            'gamma': gamma,
            'Re': Re, 
            'Fr': Fr, 
            'epsilon': epsilon,
            'Pe': Pe
            }

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

class RasingBubbleMaterial(CNodeType):
    TITLE: str = "上升气泡流体材料属性"
    PATH: str = "examples.CFD"
    INPUT_SLOTS = [
        PortConf("rho0", DataType.FLOAT, title="第一液相密度"),
        PortConf("rho1", DataType.FLOAT, title="第二液相密度"),
        PortConf("mu0", DataType.FLOAT, title="第一液相粘度"),
        PortConf("mu1", DataType.FLOAT, title="第二液相粘度"),
        PortConf("lam", DataType.FLOAT, title="应力系数"),
        PortConf("gamma", DataType.FLOAT, title="迁移率参数"),
    ]
    OUTPUT_SLOTS = [
        PortConf("material", DataType.LIST, title="物理属性")
    ]
    @staticmethod
    def run(rho0, rho1, mu0, mu1, lam, gamma):
        from fealpy.backend import backend_manager as bm
        pass
        # return material

