from ..nodetype import CNodeType, PortConf, DataType

class PointSourceMaxwellFDTDModel(CNodeType):
    r"""Point source Maxwell equations FDTD solver.

    Inputs:
        eps (float): Relative permittivity.
        mu (float): Relative permeability.
        domain (list): Computational domain [xmin, xmax, ymin, ymax] or [xmin, xmax, ymin, ymax, zmin, zmax].
        source_config (dict): Source configuration dictionary.
        object_configs (list): List of object configurations.
        n (int): Grid resolution (number of cells in each direction).
        dt (float): Time step size (optional, auto-calculated if not provided).
        maxstep (int): Maximum number of time steps.
        save_every (int): Save field data every N steps.
        boundary (str): Boundary condition type ('PEC' or 'UPML').
        pml_width (int): PML layer width for UPML boundary.
        pml_m (float): PML grading parameter.

    Outputs:
        field_history (list): List of field snapshots over time.
        fdtd_model (object): Configured FDTD model instance.
    """
    TITLE: str = "点源Maxwell方程FDTD求解器"
    PATH: str = "求解器.电磁场"
    DESC: str = """该节点实现点源Maxwell方程的FDTD（时域有限差分）求解，基于物理参数、源配置和物体配置，
                支持2D和3D计算域，可以设置边界条件和仿真参数，输出电磁场随时间演化的历史数据。"""
    
    INPUT_SLOTS = [
        # 物理参数
        PortConf("eps", DataType.FLOAT,0, title="相对介电常数"),
        PortConf("mu", DataType.FLOAT,0, title="相对磁导率"),
        PortConf("mesh", DataType.MESH, 1, title="Yee网格"),
        
        # 配置参数
        PortConf("source_config", DataType.TEXT, title="源配置"),
        PortConf("object_configs", DataType.TEXT, title="物体配置列表"),
        
        
        # FDTD仿真参数
        PortConf("dt", DataType.FLOAT, 0, title="时间步长", default=None,
                desc="时间步长，None表示自动计算"),
        PortConf("maxstep", DataType.INT, 0, title="最大时间步数", default=1000),
        PortConf("save_every", DataType.INT, 0, title="保存间隔", default=10,
                desc="每N个时间步保存一次场数据"),
        PortConf("boundary", DataType.MENU, 0, title="边界条件", default="PEC",
                items=["PEC", "UPML"], desc="边界条件类型"),
        PortConf("pml_width", DataType.INT, 0, title="PML宽度", default=8,
                desc="UPML边界层的网格点数"),
        PortConf("pml_m", DataType.FLOAT, 0, title="PML参数m", default=5.0,
                desc="UPML grading参数"),
    ]
    
    OUTPUT_SLOTS = [
        PortConf("field_history", DataType.TEXT, title="场演化历史"),
    ]

    @staticmethod
    def run(eps, mu, mesh, source_config, object_configs, 
            dt, maxstep, save_every, boundary, pml_width, pml_m):
        from fealpy.cem.point_source_maxwell_fdtd_model import PointSourceMaxwellFDTDModel
        from fealpy.cem.model.point_source_maxwell import PointSourceMaxwell
        
        domain = mesh.mesh.extent
        # 创建PDE模型
        pde = PointSourceMaxwell(eps=eps, mu=mu, domain=domain)
        
        # 添加源配置
        if source_config:
            # 如果source_config是单个源的字典，转换为列表
            if isinstance(source_config, dict):
                sources = [source_config]
            else:
                sources = source_config
            
            for src in sources:
                pde.add_source(
                    position=src.get('position'),
                    comp=src.get('comp'),
                    waveform=src.get('waveform'),
                    waveform_params=src.get('waveform_params', {}),
                    amplitude=src.get('amplitude', 1.0),
                    spread=src.get('spread', 0),
                    injection=src.get('injection', 'soft'),
                    tag=src.get('tag')
                )
        
        # 添加物体配置
        if object_configs:
            for obj in object_configs:
                pde.add_object(
                    box=obj.get('box'),
                    eps=obj.get('eps'),
                    mu=obj.get('mu'),
                    tag=obj.get('tag')
                )
        
        # 配置FDTD选项
        options = {
            'dt': dt,
            'maxstep': maxstep,
            'save_every': save_every,
            'boundary': boundary,
            'pml_width': pml_width,
            'pml_m': pml_m
        }
        n = mesh.nx
        # 创建并运行FDTD模型
        fdtd_model = PointSourceMaxwellFDTDModel(pde, n, options)
        field_history = fdtd_model.run(nt=maxstep, save_every=save_every)

        return field_history