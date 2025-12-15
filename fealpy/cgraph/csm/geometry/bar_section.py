from ...nodetype import CNodeType, PortConf, DataType


__all__ = ["UniformSection",
           "TrussTowerSection"]


class UniformSection(CNodeType):
    r"""Uniform cross-section for all bar elements.

    All bar elements have the same cross-sectional area.
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

class TrussTowerSection(CNodeType):
    r"""Truss tower cross-section calculator (dual section type).
    
    Calculates cross-sectional areas for truss tower with two types of bars:
    - Vertical bars: automatically detected based on bar orientation (z_component > 0.95)
    - Other bars: diagonal braces and horizontal bars
    
    Uses the same logic as TrussTowerData3D.bar_sections() method.
    Section area formula: A = π(do² - di²) / 4
    
    Inputs:
        mesh (mesh): Truss tower mesh.
        dov (float): Outer diameter of vertical rods [m].
        div (float): Inner diameter of vertical rods [m].
        doo (float): Outer diameter of other rods [m].
        dio (float): Inner diameter of other rods [m].
        
    Outputs:
        area (tensor): Cross-sectional area array [m²].
        is_vertical (tensor): Boolean array indicating vertical bars.
    """
    TITLE: str = "桁架塔截面属性"
    PATH: str = "examples.CSM"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.TENSOR, 1, 
                 desc="桁架塔网格", title="网格"),
        PortConf("dov", DataType.FLOAT, 0, 
                 desc="竖向杆件的外径 [m]", 
                 title="竖杆外径", default=0.015),
        PortConf("div", DataType.FLOAT, 0, 
                 desc="竖向杆件的内径 [m]", 
                 title="竖杆内径", default=0.010),
        PortConf("doo", DataType.FLOAT, 0, 
                 desc="其他杆件的外径 [m]", 
                 title="其他杆外径", default=0.010),
        PortConf("dio", DataType.FLOAT, 0, 
                 desc="其他杆件的内径 [m]", 
                 title="其他杆内径", default=0.007)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("area", DataType.TENSOR, 
                 desc="各杆件的横截面面积数组", 
                 title="横截面面积"),
        PortConf("is_vertical", DataType.TENSOR, 
                 desc="标识竖向杆件的布尔数组", 
                 title="竖杆标识")
    ]

    @staticmethod
    def run(**options):
        from fealpy.backend import backend_manager as bm
        mesh = options.get("mesh")
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        
        dov = options.get("dov")
        div = options.get("div")
        doo = options.get("doo")
        dio = options.get("dio")
        
        NC = cell.shape[0]
        Av = bm.pi * (dov**2 - div**2) / 4  # Vertical bars area
        Ao = bm.pi * (doo**2 - dio**2) / 4  # Other bars area
        
        # Calculate bar vectors
        bar_vectors = node[cell[:, 1]] - node[cell[:, 0]]  # (NC, 3)
        
        # Calculate bar lengths and unit vectors
        bar_length = bm.linalg.norm(bar_vectors, axis=1, keepdims=True)  # (NC, 1)
        unit_vectors = bar_vectors / (bar_length + 1e-12)  # (NC, 3)
        
        # Detect vertical bars based on orientation
        z_component = bm.abs(unit_vectors[:, 2])  # (NC,)
        xy_component = bm.sqrt(unit_vectors[:, 0]**2 + unit_vectors[:, 1]**2)  # (NC,)
        
        # A bar is vertical if: (|cos(θ)| > 0.95) & (|sin(θ)| < 0.3)
        is_vertical = (z_component > 0.95) & (xy_component < 0.3)
        
        # Assign cross-sectional areas
        area = bm.zeros(NC, dtype=bm.float64)
        area[is_vertical] = Av  # Vertical columns
        area[~is_vertical] = Ao  # Diagonal braces and horizontal bars
        
        return area, is_vertical