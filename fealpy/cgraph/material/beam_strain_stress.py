from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["ChannelStrainStress", 
           "TimoAxleStrainStress"]

class ChannelStrainStress(CNodeType):
    r"""Compute Strain and Stress for Channel Beam.
    
        Inputs:
            mu_y (FLOAT): Ratio of maximum to average shear stress for y-direction shear.
            mu_z (FLOAT): Ratio of maximum to average shear stress for z-direction shear.
            beam_E (FLOAT): Elastic modulus of the beam material.
            beam_nu (FLOAT): Poisson’s ratio of the beam material.
            mesh (MESH): Mesh containing node and cell information.
            uh (TENSOR): Post-processed displacement vector.
            y (FLOAT): Local coordinates in the beam cross-section.
            z (FLOAT): Local coordinates in the beam cross-section.

        Outputs:
            strain (TENSOR): Computed strain at specified locations.
            stress (TENSOR): Computed stress at specified locations.
    """
    TITLE: str = "槽形梁应变-应力计算"
    PATH: str = "material.solid"
    DESC: str = """该节点用于计算槽形梁在给定载荷和边界条件下的应变和应力。"""
    INPUT_SLOTS = [
        PortConf("mu_y", DataType.FLOAT, 1, desc="y方向剪切应力的最大值与平均值比例因子", 
                 title="y向剪切因子"),        
        PortConf("mu_z", DataType.FLOAT, 1, desc="z方向剪切应力的最大值与平均值比例因子", 
                 title="z向剪切因子"),
        PortConf("E", DataType.FLOAT, 1, desc="梁的弹性模量", title="梁的弹性模量"),
        PortConf("nu", DataType.FLOAT, 1, desc="梁的泊松比", title="梁的泊松比"),
        PortConf("mesh", DataType.MESH, 1, desc="槽形梁的三维网格", title="网格"),
        PortConf("uh", DataType.TENSOR, 1, desc="有限元分析得到的位移解向量", title="全局位移"),
        PortConf("coord_transform", DataType.TENSOR, 1, desc="坐标变换矩阵", title="坐标变换"),
        PortConf("y", DataType.FLOAT, 0, desc="应变/应力评估的y坐标", title="y坐标", default=0.0),
        PortConf("z", DataType.FLOAT, 0, desc="应变/应力评估的z坐标", title="z坐标", default=0.0),
    ]
    OUTPUT_SLOTS = [
        PortConf("strain", DataType.TENSOR, title="应变"),
        PortConf("stress", DataType.TENSOR, title="应力")
    ]
    
    @staticmethod
    def run(**options):
        
        from fealpy.csm.model.beam.channel_beam_data_3d import ChannelBeamData3D
        from fealpy.csm.material import TimoshenkoBeamMaterial
        
        mu_y = options.get("mu_y")
        mu_z = options.get("mu_z")

        model = ChannelBeamData3D(mu_y=mu_y, mu_z=mu_z)
        
        E = options.get("E")
        nu = options.get("nu")
        material = TimoshenkoBeamMaterial(model=model, 
                                        name="Timoshenko_beam",
                                        elastic_modulus=E,
                                        poisson_ratio=nu)
        
        mesh = options.get("mesh")
        disp = options.get("uh")
        uh = disp.reshape(-1, 2*model.GD)
        
        y = options.get("y", 0.0)
        z = options.get("z", 0.0)
        R = options.get("coord_transform")
        
        strain, stress = material.compute_strain_and_stress(
            mesh=mesh,
            disp=uh,
            cross_section_coords=(y, z),
            axial_position=None,
            coord_transform=R,
            ele_indices=None
        )
        
        return strain, stress


class TimoAxleStrainStress(CNodeType):
    r"""compute Strain and Stress for train-axle model.
    
        Inputs:
            beam_para (TENSOR): Beam section parameters, each row represents [Diameter, Length, Count].
            axle_para (TENSOR): Axle section parameters, each row represents [Diameter, Length, Count].
            beam_E (FLOAT): Elastic modulus of the beam material.
            beam_nu (FLOAT): Poisson’s ratio of the beam material.
            axle_E (FLOAT): Elastic modulus of the axle material.
            axle_nu (FLOAT): Poisson’s ratio of the axle material.
            mesh (MESH): Mesh containing node and cell information.
            uh (TENSOR): Post-processed displacement vector.
            y (FLOAT): Local coordinates in the beam cross-section.
            z (FLOAT): Local coordinates in the beam cross-section.
            axial_position (FLOAT)): Evaluation position along the beam axis ∈ [0, L].
                If None, the value is evaluated at the element midpoint L/2
            beam_num (INT): Number of beam elements. If None, uses all cells.
            axle_num (INT): Number of axle elements. If None, uses all cells.

        Outputs:
            beam_strain (TENSOR): Strain of the beam elements.
            beam_stress (TENSOR): Stress of the beam elements.
            axle_strain (TENSOR): Strain of the axle elements.
            axle_stress (TENSOR): Stress of the axle elements.
            strain (TENSOR): Strain of the bar elements.
            stress (TENSOR): Stress of the bar elements.

    """
    TITLE: str = "列车轮轴应变-应力计算"
    PATH: str = "material.solid"
    DESC: str = """该节点用于计算列车轮轴系统中梁段（Beam）与轴段（Axle）杆件在受力后的单元应变与应力。
            节点基于线弹性理论，通过网格、材料参数和位移场，分别对梁段与轴段进行应变–应力评价，
            适用于轮轴结构后处理、强度校核及安全性分析。"""
            
    INPUT_SLOTS = [
        PortConf("beam_para", DataType.TENSOR, 1, desc="梁结构参数数组，每行为 [直径, 长度, 数量]", title="梁段参数"),
        PortConf("axle_para", DataType.TENSOR, 1, desc="轴结构参数数组，每行为 [直径, 长度, 数量]", title="轴段参数"),
        PortConf("beam_E", DataType.FLOAT, 1, desc="梁段的弹性模量", title="梁的弹性模量"),
        PortConf("beam_nu", DataType.FLOAT, 1, desc="梁段的泊松比", title="梁的泊松比"),
        PortConf("axle_E", DataType.FLOAT, 1, desc="轴段的弹性模量", title="弹簧的弹性模量"),
        PortConf("axle_nu", DataType.FLOAT, 1, desc="轴段的泊松比", title="弹簧的泊松比"),
        PortConf("mesh", DataType.MESH, 1, desc="包含节点和单元信息的网格", title="网格"),
        PortConf("uh", DataType.TENSOR, 1, desc="未经后处理的位移向量", title="位移向量"),
        PortConf("y", DataType.FLOAT, 0, desc="截面局部 Y 坐标", title="Y坐标", default=0.0),
        PortConf("z", DataType.FLOAT, 0, desc="截面局部 Z 坐标", title="Z坐标", default=0.0),
        PortConf("axial_position", DataType.INT, 0, desc="轴向评估位置", title="轴向位置", default=None),
        PortConf("beam_num", DataType.INT, 0, desc="梁单元个数", title="梁单元", default=22),
        PortConf("axle_num", DataType.INT, 0, desc="弹簧单元个数", title="弹簧单元", default=10),
    ]
    
    OUTPUT_SLOTS = [
        PortConf("beam_strain", DataType.TENSOR, title="梁应变"),
        PortConf("beam_stress", DataType.TENSOR, title="梁应力"),
        PortConf("axle_strain", DataType.TENSOR, title="轴应变"),
        PortConf("axle_stress", DataType.TENSOR, title="轴应力"),
        PortConf("strain", DataType.TENSOR, title="应变"),
        PortConf("stress", DataType.TENSOR, title="应力")
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.backend import backend_manager as bm
        from fealpy.csm.model.beam.timobeam_axle_data_3d import TimobeamAxleData3D
        from fealpy.csm.material import TimoshenkoBeamMaterial
        from fealpy.csm.material import BarMaterial
        
        mesh = options.get("mesh")
        NC = mesh.number_of_cells()
        
        uh = options.get("uh").reshape(-1, 6)
        
        model = TimobeamAxleData3D(
                beam_para=options.get("beam_para"),
                axle_para=options.get("axle_para")
            )
        
        beam_material = TimoshenkoBeamMaterial(model=model, 
                                        name="Timo_beam",
                                        elastic_modulus=options.get("beam_E"),
                                        poisson_ratio=options.get("beam_nu"))

        beam_num = options.get("beam_num")
        beam_indices = bm.arange(0, beam_num)

        R = model.coord_transform(index=beam_indices)
        
        y = options.get("y")
        z = options.get("z")
        axial_position = options.get("axial_position")
        
        beam_strain, beam_stress = beam_material.compute_strain_and_stress(
                        mesh,
                        uh,
                        cross_section_coords=(y, z),
                        axial_position=axial_position,
                        coord_transform=R,
                        ele_indices=beam_indices)

        axle_num = options.get("axle_num")
        axle_indices = bm.arange(NC-axle_num, NC)

        axle_material = BarMaterial(
            model=model,
            name="bar",
            elastic_modulus=options.get("axle_E"),
            poisson_ratio=options.get("axle_nu")
        )
        
        axle_strain, axle_stress = axle_material.compute_strain_and_stress(
                        mesh,
                        uh,
                        ele_indices=axle_indices)
        
        strain = bm.zeros((NC, 3), dtype=bm.float64)
        stress = bm.zeros((NC, 3), dtype=bm.float64)
        
        strain[beam_indices] = beam_strain
        stress[beam_indices] = beam_stress
        
        strain[axle_indices] = axle_strain
        stress[axle_indices] = axle_stress
        
        return (beam_strain, beam_stress,
                axle_strain, axle_stress, 
                strain, stress, )