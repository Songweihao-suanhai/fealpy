from ...nodetype import CNodeType, PortConf, DataType


__all__ = ["UniformSection"]


class UniformSection(CNodeType):
    r"""Uniform cross-section for all elements.

    All elements have the same cross-sectional area.
    Supports different geometric shapes: rectangular, circular, and tubular.
    
    Inputs:
        shape_type (MENU): Geometric shape of the cross-section.
        width (float): Width of rectangle [m] (for rectangular shape).
        height (float): Height of rectangle [m] (for rectangular shape).
        diameter (float): Diameter of circle [m] (for circular shape).
        outer_diameter (float): Outer diameter of tube [m] (for tubular shape).
        inner_diameter (float): Inner diameter of tube [m] (for tubular shape).
        num_elements (int): Total number of elements.
        
    Outputs:
        area (tensor): Cross-sectional area array [m²].
    """
    TITLE: str = "常见截面属性"
    PATH: str = "examples.CSM"
    INPUT_SLOTS = [
        PortConf("shape_type", DataType.MENU, 0, 
                 desc="截面的几何形状", 
                 title="截面形状", 
                 default="rectangular", 
                 items=["rectangular", "circular", "tubular"]),
        PortConf("width", DataType.FLOAT, 0, 
                 desc="矩形截面的宽度 [m]", 
                 title="宽度", default=0.01),
        PortConf("height", DataType.FLOAT, 0, 
                 desc="矩形截面的高度 [m]", 
                 title="高度", default=0.01),
        PortConf("diameter", DataType.FLOAT, 0, 
                 desc="圆形截面的直径 [m]", 
                 title="直径", default=0.01),
        PortConf("outer_diameter", DataType.FLOAT, 0, 
                 desc="圆管截面的外径 [m]", 
                 title="外径", default=0.01),
        PortConf("inner_diameter", DataType.FLOAT, 0, 
                 desc="圆管截面的内径 [m]", 
                 title="内径", default=0.008)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("area", DataType.FLOAT, 
                 desc="统一的横截面面积值 [m²]", 
                 title="横截面面积")
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import backend_manager as bm
        
        shape_type = options.get("shape_type")
        num = options.get("num_elements")
        
        if shape_type == "rectangular":
            width = options.get("width")
            height = options.get("height")
            area = width * height
            
        elif shape_type == "circular":
            diameter = options.get("diameter")
            area = bm.pi * diameter**2 / 4
            
        elif shape_type == "tubular":
            do = options.get("outer_diameter")
            di = options.get("inner_diameter")
            area = bm.pi * (do**2 - di**2) / 4
            
        else:
            raise ValueError(f"Unsupported shape_type: {shape_type}")
        
        return area