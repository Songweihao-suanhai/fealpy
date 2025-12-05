from ..nodetype import CNodeType, PortConf, DataType

class CHNSEquation(CNodeType):
    TITLE: str = "Cahn–Hilliard–Navier–Stokes 方程"
    PATH: str = "preprocess.equations"
    INPUT_SLOTS = [
        PortConf("rho", DataType.FUNCTION, title="密度"),
        PortConf("Re", DataType.FLOAT, title="雷诺数"),
        PortConf("Fr", DataType.FLOAT, title="弗劳德数"),
        PortConf("epsilon", DataType.FLOAT, title="界面厚度参数"),
        PortConf("Pe", DataType.FLOAT, title="Peclet 数"),
        PortConf("body_force", DataType.FUNCTION, title="源项"),
    ]
    OUTPUT_SLOTS = [
        PortConf("mobility", DataType.FLOAT, desc="相场迁移率", title="迁移率"),
        PortConf("interface", DataType.FLOAT, desc="界面参数", title="界面参数"),
        PortConf("free_energy", DataType.FLOAT, desc="自由能参数", title="自由能参数"),
        PortConf("time_derivative", DataType.FUNCTION, desc="NS时间项系数函数", title="时间项系数"),
        PortConf("convection", DataType.FUNCTION, desc="对流项系数函数", title="对流项系数"),
        PortConf("pressure", DataType.FUNCTION, desc="压力项系数函数", title="压力项系数"),
        PortConf("viscosity", DataType.FUNCTION, desc="粘性项系数函数", title="粘性项系数"),
        PortConf("source", DataType.FUNCTION, desc="源项函数", title="源项函数"),
    ]
    @staticmethod
    def run(rho, Re, Fr, epsilon, Pe, body_force):
        mobility = 1/Pe
        interface = epsilon ** 2
        free_energy = 1
        time_derivative = rho
        convection = rho
        pressure = 1
        viscosity = 1/Re
        source = body_force
        return (mobility, interface, free_energy, time_derivative,
                convection, pressure, viscosity, source)