from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["Bar25Material", "Bar942Material", "TrussTowerMaterial",
           "AxleMaterial"]

 
class Bar25Material(CNodeType):
    r"""A CGraph node for defining the material properties of a 3D truss/bar.

    Inputs:
        property(MENU): Material name, e.g."structural-steel".
        type(STRING): Type of bar material.
        E(FLOAT): Elastic modulus of the bar material.
        nu(FLOAT): Poisson's ratio of the bar material.

    Outputs:
        E (float): The elastic modulus of the bar material.
        nu(FLOAT): Poisson's ratio of the bar material.

    """
    TITLE: str = "25杆的材料属性"
    PATH: str = "material.solid"
    DESC: str = "定义25杆的线弹性材料属性"
    INPUT_SLOTS = [
         PortConf("property", DataType.MENU, 0, desc="材料名称", title="材料材质", default="structural-steel", 
                 items=["structural-steel", "aluminum", "concrete", "plastic", "wood", "alloy"]),
        PortConf("type", DataType.STRING, 0, desc="材料类型", title="杆件类型", default="bar25"),
        PortConf("E", DataType.FLOAT, 0, desc="杆的弹性模量", title="杆弹性模量", default=1500),
        PortConf("nu", DataType.FLOAT, 0, desc="杆的泊松比", title="杆泊松比", default=0.3)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("E", DataType.FLOAT, title="杆的弹性模量"),
        PortConf("nu", DataType.FLOAT, title="杆的泊松比"),
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.csm.model.truss.bar_data25_3d import BarData25
        from fealpy.csm.material import BarMaterial

        model = BarData25()
        material = BarMaterial(
            model=model,
            name=options.get("type"),
            elastic_modulus=options.get("E"),
            poisson_ratio=options.get("nu")
        )

        return tuple(
            getattr(material, name) for name in ["E", "nu"]
        )


class Bar942Material(CNodeType):
    r"""A CGraph node for defining the material properties of a 3D truss/bar.

    Inputs:
        property(MENU): Material name, e.g."structural-steel".
        type(STRING): Type of bar material.
        E(FLOAT): Elastic modulus of the bar material.
        nu(FLOAT): Poisson's ratio of the bar material.

    Outputs:
        E (float): The elastic modulus of the bar material.
        nu(FLOAT): Poisson's ratio of the bar material.
    """
    TITLE: str = "942杆的材料属性"
    PATH: str = "material.solid"
    DESC: str = "定义942杆的线弹性材料属性"
    INPUT_SLOTS = [
         PortConf("property", DataType.MENU, 0, desc="材料名称", title="材料材质", default="structural-steel", 
                 items=["structural-steel", "aluminum", "concrete", "plastic", "wood", "alloy"]),
        PortConf("type", DataType.STRING, 0, desc="材料类型", title="杆件类型", default="bar942"),
        PortConf("E", DataType.FLOAT, 0, desc="杆的弹性模量", title="杆弹性模量", default=2.1e5),
        PortConf("nu", DataType.FLOAT, 0, desc="杆的泊松比", title="杆泊松比", default=0.3)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("E", DataType.FLOAT, title="杆的弹性模量"),
        PortConf("nu", DataType.FLOAT, title="杆的泊松比"),
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.csm.model.truss.bar_data942_3d import BarData942
        from fealpy.csm.material import BarMaterial

        model = BarData942()
        material = BarMaterial(
            model=model,
            name=options.get("type"),
            elastic_modulus=options.get("E"),
            poisson_ratio=options.get("nu")
        )

        return tuple(
            getattr(material, name) for name in ["E", "nu"]
        )


class TrussTowerMaterial(CNodeType):
    r"""truss Material Definition Node.
    
        Inputs:
            property (string): Material name, e.g., "Steel".
            type (string): Type of axle material.
            dov (float): Outer diameter of vertical rods (m).
            div (float): Inner diameter of vertical rods (m).
            doo (float): Outer diameter of other rods (m).
            dio (float): Inner diameter of other rods (m).
            E (float): Elastic modulus of the axle material.
            nu (float): Poisson’s ratio of the axle material.
 
        Outputs:
            E (float): Elastic modulus of the axle material.
            mu (float): Shear modulus, computed as `E / [2(1 + nu)]`.
    
    """
    TITLE: str = "桁架塔材料属性"
    PATH: str = "material.solid"
    DESC: str = """该节点定义桁架塔结构中所使用的杆件材料参数，包括材料名称、类型、弹性模量与泊松比等内容，
            通过标准化的材料属性输出，为有限元单元刚度计算与结构分析提供统一的材料基础。"""
            
    INPUT_SLOTS = [
        PortConf("property", DataType.MENU, 0, desc="材料名称", title="材料材质", default="structural-steel", 
                 items=["structural-steel", "aluminum", "concrete", "plastic", "wood", "alloy"]),
        PortConf("type", DataType.STRING, 0, desc="材料类型", title="杆件类型", default="bar"),
        PortConf("E", DataType.FLOAT, 0, desc="杆件的弹性模量", title="弹性模量", default=2.0e11),
        PortConf("nu", DataType.FLOAT, 0, desc="杆件的泊松比", title="泊松比", default=0.3)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("E", DataType.FLOAT, title="弹性模量"),
        PortConf("nu", DataType.FLOAT, title="泊松比"),
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.csm.model.truss.truss_tower_data_3d import TrussTowerData3D
        from fealpy.csm.material import BarMaterial
        
        model = TrussTowerData3D()
        material = BarMaterial(
            model=model,
            name=options.get("type"),
            elastic_modulus=options.get("E"),
            poisson_ratio=options.get("nu")
        )

        return tuple(
            getattr(material, name) for name in ["E", "nu"]
        )
        

class AxleMaterial(CNodeType):
    r"""Axle Material Definition Node.
    Inputs:
        property (STRING): Material name, e.g., "Steel".
        axle_type (MENU): Type of axle material.
        beam_para (TENSOR): Beam section parameters, each row represents [Diameter, Length, Count].
        axle_para (TENSOR): Axle section parameters, each row represents [Diameter, Length, Count].
        axle_stiffness (FLOAT): spring stiffness.
        axle_E (FLOAT): Elastic modulus of the axle material.
        axle_nu (FLOAT): Poisson’s ratio of the axle material.

        Outputs:
            E (FLOAT): Elastic modulus of the axle material.
            nu (FLOAT): Poisson’s ratio of the axle material.

    """
    TITLE: str = "列车轮轴弹簧材料属性"
    PATH: str = "material.solid"
    DESC: str = """该节点计算列车轮轴系统中杆件（Spring / Axle-Rod）部分的材料属性，
        包括弹性模量、泊松比和剪切模量。适用于需要对轴上的弹簧、连接杆等结构进行建模的场景。"""
        
    INPUT_SLOTS = [
        PortConf("property", DataType.STRING, 0, desc="材料名称（如钢、铝等）", title="材料材质", default="Steel"),
        PortConf("axle_type", DataType.STRING, 0, desc="轮轴材料类型", title="弹簧材料", default="Spring"),
        PortConf("beam_para", DataType.TENSOR, 1, desc="梁结构参数数组，每行为 [直径, 长度, 数量]", title="梁段参数"),
        PortConf("axle_para", DataType.TENSOR, 1, desc="轴结构参数数组，每行为 [直径, 长度, 数量]", title="轴段参数"),
        PortConf("axle_stiffness", DataType.FLOAT, 0, desc="弹簧刚度", title="弹簧的刚度", default=1.976e6),
        PortConf("axle_E", DataType.FLOAT, 0, desc="弹簧弹性模量", title="弹簧的弹性模量", default=1.976e6),
        PortConf("axle_nu", DataType.FLOAT, 0, desc="弹簧泊松比", title="弹簧的泊松比", default=-0.5)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("E", DataType.FLOAT, title="弹簧的弹性模量"),
        PortConf("nu", DataType.FLOAT, title="弹簧的泊松比")
    ]
    
    @staticmethod
    def run(property, axle_type, axle_stiffness, 
            beam_para=None, axle_para=None, 
            axle_E=1.976e6, axle_nu=-0.5):
        from fealpy.csm.model.beam.timobeam_axle_data_3d import TimobeamAxleData3D
        from fealpy.csm.material import BarMaterial
        
        model = TimobeamAxleData3D(beam_para, axle_para)
        
        axle_material = BarMaterial(model=model,
                                name=axle_type,
                                elastic_modulus=axle_E,
                                poisson_ratio=axle_nu)
        return tuple(
            getattr(axle_material, name) for name in ["E", "nu"]
        )