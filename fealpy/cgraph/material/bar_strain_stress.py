from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["BarStrainStress"]

class BarStrainStress(CNodeType):
    r"""compute Strain and Stress for Bar Elements.
    
    Inputs:
        mesh (MESH): Mesh containing node and cell information.
        uh (TENSOR): Post-processed displacement vector.
        coord_transform (TENSOR): Coordinate transformation matrix.

    Outputs:
        strain (TENSOR): Strain of the bar elements.
        stress (TENSOR): Stress of the bar elements.

    """
    TITLE: str = "杆单元应变-应力计算"
    PATH: str = "material.solid"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, desc="包含节点和单元信息的网格", title="网格"),
        PortConf("uh", DataType.TENSOR, 1, desc="未经后处理的位移", title="全局位移"),
        PortConf("coord_transform", DataType.TENSOR, 1, desc="坐标变换矩阵", title="坐标变换")
    ]
    
    OUTPUT_SLOTS = [
        PortConf("strain", DataType.TENSOR, title="应变"),
        PortConf("stress", DataType.TENSOR, title="应力")
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.csm.material import BarMaterial
        
        mesh = options.get("mesh")
        
        E_scalar = float(mesh.celldata['E'][0])
        nu_scalar = float(mesh.celldata['nu'][0])
        
        material = BarMaterial(
            model=None,
            name="bar",
            elastic_modulus=E_scalar,
            poisson_ratio=nu_scalar
        )
        
        uh = options.get("uh").reshape(-1, 3)
        strain, stress = material.compute_strain_and_stress(
                        options.get("mesh"),
                        uh,
                        options.get("coord_transform"),
                        ele_indices=None)

        return strain, stress