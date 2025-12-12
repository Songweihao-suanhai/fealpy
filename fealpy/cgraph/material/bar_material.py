from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["BarMaterial", 
           "AxleMaterial"]

 
class BarMaterial(CNodeType):
    r"""Material property definition for truss/bar structures.
    
    Supports two input modes:
    1. Predefined case (bar25/bar942/truss_tower): Uses standard material parameters
    2. Custom: Choose material type (property) or input custom E, nu
    
    Inputs:
        bar_type (MENU): Bar structure type.
        property (MENU): Material type (for custom bar_type).
        E (float): Elastic modulus [Pa] (for custom property).
        nu (float): Poisson's ratio (for custom property).

    Outputs:
        E (float): Elastic modulus [Pa].
        nu (float): Poisson's ratio.
    """
    TITLE: str = "杆材料属性"
    PATH: str = "material.solid"
    INPUT_SLOTS = [
        PortConf("bar_type", DataType.MENU, 0, 
                 desc="杆件结构类型", 
                 title="结构类型", 
                 default="custom",
                 items=["bar25", "bar942", "truss_tower", "custom"]),
        PortConf("property", DataType.MENU, 0, 
                 desc="材料材质类型 (仅custom结构类型有效)", 
                 title="材料材质", 
                 default="custom-input", 
                 items=["structural-steel", "aluminum", "concrete", "titanium", "brass", "custom-input"]),
        PortConf("E", DataType.FLOAT, 0, 
                 desc="弹性模量 [Pa] (仅custom材质有效)", 
                 title="弹性模量", 
                 default=2.0e11),
        PortConf("nu", DataType.FLOAT, 0, 
                 desc="泊松比 (仅custom材质有效)", 
                 title="泊松比", 
                 default=0.3)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("E", DataType.FLOAT, title="弹性模量"),
        PortConf("nu", DataType.FLOAT, title="泊松比")
    ]
    
    @staticmethod
    def run(**options):
        bar_type = options.get("bar_type")
        property_type = options.get("property")
        E = options.get("E")
        nu = options.get("nu")
        
        # 第一层：案例预设参数
        case_presets = {
            "bar25": {"E": 1500, "nu": 0.3},           
            "bar942": {"E": 2.1e5, "nu": 0.3},         
            "truss_tower": {"E": 2.0e11, "nu": 0.3}    
        }
        
        if bar_type in case_presets:
            material = case_presets[bar_type]
            return (material["E"], material["nu"])
        
        # 第二层：材质数据库（用于custom结构类型）
        material_database = {
            # 金属材料
            "structural-steel": {"E": 2.0e11, "nu": 0.3},
            "stainless-steel": {"E": 1.93e11, "nu": 0.31},
            "aluminum": {"E": 7.0e10, "nu": 0.33},
            "aluminum-alloy": {"E": 7.2e10, "nu": 0.33},
            "titanium": {"E": 1.1e11, "nu": 0.34},
            "titanium-alloy": {"E": 1.14e11, "nu": 0.34},
            "brass": {"E": 1.0e11, "nu": 0.34},
            "copper": {"E": 1.2e11, "nu": 0.34},
            
            # 非金属材料
            "concrete": {"E": 3.0e10, "nu": 0.2},
            "wood": {"E": 1.0e10, "nu": 0.35},
            
            # 复合材料
            "carbon-fiber": {"E": 1.5e11, "nu": 0.3},
            "fiberglass": {"E": 3.5e10, "nu": 0.25}
        }
        
        # 如果选择了预定义材质
        if property_type in material_database:
            material = material_database[property_type]
            return (material["E"], material["nu"])
        return E, nu


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