from typing import Union, List, Dict, Any, Tuple, Optional
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["PointSourceMaxwell"]

class PointSourceMaxwell(CNodeType):
    r"""Point source Maxwell equations problem model.

    Inputs:
        domain (list): Computational domain [xmin, xmax, ymin, ymax] or [xmin, xmax, ymin, ymax, zmin, zmax].
        eps (float): Relative permittivity.
        mu (float): Relative permeability.
        
        # Single source configuration
        source_position (list): Source position coordinates.
        source_component (str): Source field component.
        source_waveform (str): Source waveform type.
        source_amplitude (float): Source amplitude.
        source_spread (int): Source spatial spread.
        source_injection (str): Source injection method.
        
        # Two object configurations
        object1_box (list): First object bounding box.
        object1_eps (float): First object relative permittivity.
        object1_mu (float): First object relative permeability.
        
        object2_box (list): Second object bounding box.
        object2_eps (float): Second object relative permittivity.
        object2_mu (float): Second object relative permeability.
    
    Outputs:
        eps (float): Relative permittivity.
        mu (float): Relative permeability.
        domain (domain): Computational domain.
        source_config (dict): Source configuration.
        object_configs (list): Object configurations.
        pde_model (object): Configured PointSourceMaxwell instance.
    """
    TITLE: str = "点源Maxwell方程问题模型"
    PATH: str = "模型.电磁场"
    DESC: str = """该节点定义点源Maxwell方程模型，支持2D和3D计算域，可以设置背景材料参数、单个点源激励和最多两个物体区域，为FDTD仿真提供物理问题定义。"""
    
    INPUT_SLOTS = [
        PortConf("domain", DataType.DOMAIN, 0, title="计算域", default=[0, 1, 0, 1]),
        PortConf("eps", DataType.FLOAT, 0, title="相对介电常数", default=1.0),
        PortConf("mu", DataType.FLOAT, 0, title="相对磁导率", default=1.0),
        
        # Single source configuration
        PortConf("source_position", DataType.DOMAIN, 0, title="源位置", default=[0.5, 0.5],
                desc="源位置坐标，格式为(x,y)或(x,y,z)"),
        PortConf("source_component", DataType.MENU, 0, title="源场分量", default="Ez",
                items=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
                desc="源激励的电磁场分量"),
        PortConf("source_waveform", DataType.MENU, 0, title="源波形类型", default="gaussian",
                items=["gaussian", "sinusoid", "ricker", "gaussian_enveloped_sine"],
                desc="源的时间波形类型"),
        PortConf("source_amplitude", DataType.FLOAT, 0, title="源幅度", default=1.0,
                desc="源的幅度大小"),
        PortConf("source_spread", DataType.INT, 0, title="源扩展半径", default=0,
                desc="源的空间扩展半径（网格点数）"),
        PortConf("source_injection", DataType.MENU, 0, title="源注入方式", default="soft",
                items=["soft", "hard"],
                desc="源的注入方式：soft(叠加)或hard(覆盖)"),
        
        # First object configuration
        PortConf("object1_box", DataType.DOMAIN, 0, title="物体1边界框", default=None,
                desc="第一个物体的边界框，格式为[xmin, xmax, ymin, ymax]或[xmin, xmax, ymin, ymax, zmin, zmax]"),
        PortConf("object1_eps", DataType.FLOAT, 0, title="物体1介电常数", default=None,
                desc="第一个物体的相对介电常数，None表示使用背景值"),
        PortConf("object1_mu", DataType.FLOAT, 0, title="物体1磁导率", default=None,
                desc="第一个物体的相对磁导率，None表示使用背景值"),
        
        # Second object configuration
        PortConf("object2_box", DataType.DOMAIN, 0, title="物体2边界框", default=None,
                desc="第二个物体的边界框，格式为[xmin, xmax, ymin, ymax]或[xmin, xmax, ymin, ymax, zmin, zmax]"),
        PortConf("object2_eps", DataType.FLOAT, 0, title="物体2介电常数", default=None,
                desc="第二个物体的相对介电常数，None表示使用背景值"),
        PortConf("object2_mu", DataType.FLOAT, 0, title="物体2磁导率", default=None,
                desc="第二个物体的相对磁导率，None表示使用背景值"),
    ]
    
    OUTPUT_SLOTS = [
        PortConf("eps", DataType.FLOAT, title="相对介电常数"),
        PortConf("mu", DataType.FLOAT, title="相对磁导率"),
        PortConf("domain", DataType.DOMAIN, title="计算域"),
        PortConf("source_config", DataType.TEXT, title="源配置"),
        PortConf("object_configs", DataType.TEXT, title="物体配置列表"),
    ]

    @staticmethod
    def run(domain: List[float], eps: float, mu: float,
            source_position: List[float], source_component: str,
            source_waveform: str, source_amplitude: float,
            source_spread: int, source_injection: str,
            object1_box: Optional[List[float]], object1_eps: Optional[float], object1_mu: Optional[float],
            object2_box: Optional[List[float]], object2_eps: Optional[float], object2_mu: Optional[float]) -> tuple:
        from fealpy.cem.model.point_source_maxwell import PointSourceMaxwell as MaxwellModel
        
        # 创建Maxwell问题模型
        model = MaxwellModel(eps=eps, mu=mu, domain=domain)
        
        # 配置波形参数（根据波形类型设置默认参数）
        waveform_params = {}
        if source_waveform == "gaussian":
            waveform_params = {"t0": 1.0, "tau": 0.2}
        elif source_waveform == "ricker":
            waveform_params = {"t0": 1.0, "f": 1.0}
        elif source_waveform == "sinusoid":
            waveform_params = {"freq": 1.0, "phase": 0.0}
        elif source_waveform == "gaussian_enveloped_sine":
            waveform_params = {"freq": 1.0, "t0": 1.0, "tau": 0.2}
        
        # 添加单个源
        source_config = {}
        if source_position and source_component:
            source_tag = model.add_source(
                position=tuple(source_position),
                comp=source_component,
                waveform=source_waveform,
                waveform_params=waveform_params,
                amplitude=source_amplitude,
                spread=source_spread,
                injection=source_injection
            )
            # 获取源配置信息
            source_cfgs = model.list_sources()
            source_config = next((s for s in source_cfgs if s['tag'] == source_tag), {})
        
        # 添加物体配置
        object_configs = []
        
        # 添加第一个物体（如果提供了边界框）
        if object1_box:
            object1_tag = model.add_object(
                box=object1_box,
                eps=object1_eps,
                mu=object1_mu,
                conductivity=0.0,  # 默认不导电
                tag="object1"
            )
            obj1_cfgs = model.list_objects()
            obj1_config = next((o for o in obj1_cfgs if o['tag'] == object1_tag), {})
            if obj1_config:
                object_configs.append(obj1_config)
        
        # 添加第二个物体（如果提供了边界框）
        if object2_box:
            object2_tag = model.add_object(
                box=object2_box,
                eps=object2_eps,
                mu=object2_mu,
                conductivity=0.0,  # 默认不导电
                tag="object2"
            )
            obj2_cfgs = model.list_objects()
            obj2_config = next((o for o in obj2_cfgs if o['tag'] == object2_tag), {})
            if obj2_config:
                object_configs.append(obj2_config)
        
        return (model.eps, model.mu, model.domain, 
                source_config, object_configs)