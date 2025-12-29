from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["ElectromagneticMaterial"]


class ElectromagneticMaterial(CNodeType):
    r"""Electromagnetic material property definition node.

    Inputs:
        property (MENU): Material type selection.
        eps (FLOAT): Relative permittivity (only effective when property='custom-input').
        mu (FLOAT): Relative permeability (only effective when property='custom-input').

    Outputs:
        mp (LIST): Material properties list containing eps and mu.
    """
    TITLE: str = "电磁材料"
    PATH: str = "material.electromagnetic"

    INPUT_SLOTS = [
        PortConf("property", DataType.MENU, 0, 
                desc="材料类型选择", 
                title="材料类型", 
                default="custom-input", 
                items=["vacuum", "air", "glass", "fused-silica", "silicon", 
                       "silicon-dioxide", "gallium-arsenide", "teflon", 
                       "water", "gold", "silver", "copper", "aluminum",
                       "perfect-conductor", "custom-input"]),
        PortConf("eps", DataType.FLOAT, 0, 
                desc="相对介电常数 (仅当材料类型为'custom-input'时有效)", 
                title="相对介电常数", 
                default=1.0),
        PortConf("mu", DataType.FLOAT, 0, 
                desc="相对磁导率 (仅当材料类型为'custom-input'时有效)", 
                title="相对磁导率", 
                default=1.0)
    ]

    OUTPUT_SLOTS = [
        PortConf("mp", DataType.DICT, title="材料属性"),
    ]

    @staticmethod
    def run(**options):
        property_type = options.get("property")

        # 电磁材料数据库
        material_database = {
            # 真空和气体
            "vacuum": {"eps": 1.0, "mu": 1.0},
            "air": {"eps": 1.00054, "mu": 1.0},

            # 常见介质材料
            "glass": {"eps": 6.0, "mu": 1.0},
            "fused-silica": {"eps": 3.8, "mu": 1.0},
            "silicon": {"eps": 11.7, "mu": 1.0},
            "silicon-dioxide": {"eps": 3.9, "mu": 1.0},
            "gallium-arsenide": {"eps": 12.9, "mu": 1.0},
            "teflon": {"eps": 2.1, "mu": 1.0},
            "water": {"eps": 80.0, "mu": 1.0},

            # 金属材料 (使用Drude模型的低频近似)
            "gold": {"eps": 1.0, "mu": 1.0},
            "silver": {"eps": 1.0, "mu": 1.0},
            "copper": {"eps": 1.0, "mu": 1.0},
            "aluminum": {"eps": 1.0, "mu": 1.0},

            # 理想导体
            "perfect-conductor": {"eps": 1.0, "mu": 1.0},
        }

        # 如果选择了预定义材料
        if property_type in material_database:
            material = material_database[property_type]
            mp = {
                'eps': material["eps"],
                'mu': material["mu"]
            }
        else:
            # 使用自定义输入
            mp = {
                'eps': options.get("eps"),
                'mu': options.get("mu")
            }

        return mp