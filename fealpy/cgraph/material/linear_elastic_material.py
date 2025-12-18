from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["LinearElasticMaterial", 
           "AxleMaterial"]

 
class LinearElasticMaterial(CNodeType):
    r"""Linear elastic material property definition node.
    
    Universal linear elastic material for all structural elements (bar, beam, solid, etc.).
    Supports two input modes:
    1. Predefined materials: Select from material database
    2. Custom input: Manually input material properties
    
    Inputs:
        Inputs:
        property (MENU): Material type selection.
        E (FLOAT): Elastic modulus [Pa] (only effective when property='custom-input').
        nu (FLOAT): Poisson's ratio (only effective when property='custom-input').
        rho (FLOAT): Density [kg/m³] (only effective when property='custom-input').

    Outputs:
        E (FLOAT): Elastic modulus [Pa].
        nu (FLOAT): Poisson's ratio.
        rho (FLOAT): Density [kg/m³].
        mu (FLOAT): Shear modulus [Pa], calculated as mu = E / (2 * (1 + nu)).
        lambda_ (FLOAT): Lamé's first parameter [Pa].
    """
    TITLE: str = "线弹性材料"
    PATH: str = "material.solid"
    INPUT_SLOTS = [
        PortConf("property", DataType.MENU, 0, 
                desc="材料类型选择", 
                title="材料类型", 
                default="structural-steel", 
                items=["structural-steel", "stainless-steel", "aluminum", "aluminum-alloy", 
                       "titanium", "titanium-alloy", "brass", "copper", "concrete", "wood", 
                       "carbon-fiber", "fiberglass", "custom-input"]),
        PortConf("E", DataType.FLOAT, 0, 
                desc="弹性模量(仅当材料类型为'custom-input'时有效)", 
                title="弹性模量", 
                default=2.0e11),
        PortConf("nu", DataType.FLOAT, 0, 
                desc="泊松比 (仅当材料类型为'custom-input'时有效)", 
                title="泊松比", 
                default=0.3),
        PortConf("rho", DataType.FLOAT, 0, 
                desc="密度 [kg/m³] (仅当材料类型为'custom-input'时有效)", 
                title="密度", 
                default=7850.0)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("E", DataType.FLOAT, title="弹性模量"),
        PortConf("nu", DataType.FLOAT, title="泊松比"),
        PortConf("rho", DataType.FLOAT, title="密度"),
        PortConf("mu", DataType.FLOAT, title="剪切模量"),
        PortConf("lambda_", DataType.FLOAT, title="拉梅常数")
    ]
    
    @staticmethod
    def run(**options):
        property_type = options.get("property")
        # 材质数据库
        material_database = {
            # 金属材料
            "structural-steel": {"E": 2.0e11, "nu": 0.3, "rho": 7850.0},
            "stainless-steel": {"E": 1.93e11, "nu": 0.31, "rho": 8000.0},
            "aluminum": {"E": 7.0e10, "nu": 0.33, "rho": 2700.0},
            "aluminum-alloy": {"E": 7.2e10, "nu": 0.33, "rho": 2800.0},
            "titanium": {"E": 1.1e11, "nu": 0.34, "rho": 4500.0},
            "titanium-alloy": {"E": 1.14e11, "nu": 0.34, "rho": 4430.0},
            "brass": {"E": 1.0e11, "nu": 0.34, "rho": 8500.0},
            "copper": {"E": 1.2e11, "nu": 0.34, "rho": 8960.0},
            
            # 非金属材料
            "concrete": {"E": 3.0e10, "nu": 0.2, "rho": 2400.0},
            "wood": {"E": 1.0e10, "nu": 0.35, "rho": 600.0},
            
            # 复合材料
            "carbon-fiber": {"E": 1.5e11, "nu": 0.3, "rho": 1600.0},
            "fiberglass": {"E": 3.5e10, "nu": 0.25, "rho": 2000.0}
        }
        
        # 如果选择了预定义材质
        if property_type in material_database:
            material = material_database[property_type]
            E = material["E"]
            nu = material["nu"]
            rho = material["rho"]
        else:
            # 使用自定义输入
            E = options.get("E")
            nu = options.get("nu")
            rho = options.get("rho")
        
        # 计算剪切模量: mu = E / (2 * (1 + nu))
        mu = E / (2.0 * (1.0 + nu))
        
        # 计算拉梅第一参数: lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
        lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        return E, nu, rho, mu, lambda_


class AxleMaterial(CNodeType):
    r"""Axle Material Definition Node.
    
    Inputs:
        property(MENU): Material name, e.g."structural-steel".
        type(STRING): Type of bar material.
        beam_para (TENSOR): Beam section parameters, each row represents [Diameter, Length, Count].
        axle_para (TENSOR): Axle section parameters, each row represents [Diameter, Length, Count].
        stiffness (FLOAT): spring stiffness.
        E (FLOAT): Elastic modulus of the axle material.
        nu (FLOAT): Poisson's ratio of the axle material.

        Outputs:
            E (FLOAT): Elastic modulus of the axle material.
            nu (FLOAT): Poisson's ratio of the axle material.
            mu (FLOAT): Shear modulus of the axle material.

    """
    TITLE: str = "列车车轴弹簧部分材料属性"
    PATH: str = "material.solid"
    INPUT_SLOTS = [
        PortConf("property", DataType.MENU, 0, desc="材料名称", title="材料材质", default="structural-steel", 
                 items=["structural-steel", "aluminum", "concrete", "plastic", "wood", "alloy"]),
        PortConf("type", DataType.STRING, 0, desc="材料类型", title="弹簧类型", default="spring"),
        PortConf("beam_para", DataType.TENSOR, 1, desc="梁结构参数数组，每行为 [直径, 长度, 数量]", title="梁段参数"),
        PortConf("axle_para", DataType.TENSOR, 1, desc="轴结构参数数组，每行为 [直径, 长度, 数量]", title="轴段参数"),
        PortConf("stiffness", DataType.FLOAT, 0, desc="弹簧刚度", title="弹簧的刚度", default=1.976e6),
        PortConf("E", DataType.FLOAT, 0, desc="弹簧弹性模量", title="弹簧的弹性模量", default=1.976e6),
        PortConf("nu", DataType.FLOAT, 0, desc="弹簧泊松比", title="弹簧的泊松比", default=-0.5)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("E", DataType.FLOAT, title="弹簧的弹性模量"),
         PortConf("nu", DataType.FLOAT, title="弹簧的泊松比"),
        PortConf("mu", DataType.FLOAT, title="弹簧的剪切模量")
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.csm.model.beam.timobeam_axle_data_3d import TimobeamAxleData3D
        from fealpy.csm.material import AxleMaterial

        model = TimobeamAxleData3D(
            beam_para=options.get("beam_para"),
            axle_para=options.get("axle_para")
        )

        axle_material = AxleMaterial(model=model,
                                name=options.get("type"),
                                elastic_modulus=options.get("E"),
                                poisson_ratio=options.get("nu")
                            )
        return tuple(
            getattr(axle_material, name) for name in ["E", "nu", "mu"]
        )