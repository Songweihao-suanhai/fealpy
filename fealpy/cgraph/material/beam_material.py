from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["BeamMaterial", "ChannelBeamMaterial"]


class BeamMaterial(CNodeType):
    r"""Euler-Bernoulli Beam Material Definition Node.
    
        Inputs:
            property (string): Material type (e.g., Steel, Aluminum).
            beam_type (menu): Beam model type selection.
            beam_E (float): Elastic modulus of the beam.
            beam_nu (float): Poisson's ratio of the beam.
            I (float): Second moment of area (area moment of inertia).
            
        Outputs:
            I (tensor): Second moment of area.
            E (float): Elastic modulus of the beam.
            nu (float): Poisson's ratio of the beam.
    """
    TITLE: str = "欧拉梁材料属性"
    PATH: str = "材料.固体"
    DESC: str = "欧拉梁材料属性"
    INPUT_SLOTS = [
        PortConf("property", DataType.STRING, 0, desc="材料名称（如钢、铝等）", title="材料材质", default="Steel"),
        PortConf("beam_type", DataType.MENU, 0, desc="梁模型类型选择", title="梁材料类型",
                 items=["Euler-Bernoulli", "Timoshenko"]),
        PortConf("beam_E", DataType.FLOAT, 0, desc="梁的弹性模量", title="梁弹性模量", default=200e9),
        PortConf("beam_nu", DataType.FLOAT, 0, desc="梁的泊松比", title="梁泊松比", default=0.3),
        PortConf("I", DataType.FLOAT, 0, desc="惯性矩", title="惯性矩", default=118.6e-6),
        
    ]
    
    OUTPUT_SLOTS = [
        PortConf("I", DataType.TENSOR, title="惯性矩"),
        PortConf("E", DataType.FLOAT, title="梁的弹性模量"),
        PortConf("nu", DataType.FLOAT, title="梁的泊松比"),
    ]
    
    @staticmethod
    def run(property="Steel", beam_type="Euler-Bernoulli beam", beam_E=200e9, beam_nu=0.3, I=118.6e-6):
        from fealpy.csm.model.beam.euler_bernoulli_beam_data_2d import EulerBernoulliBeamData2D
        from fealpy.csm.material import EulerBernoulliBeamMaterial
        model = EulerBernoulliBeamData2D()
        beam_material = EulerBernoulliBeamMaterial(model=model, 
                                        name="eulerbeam",
                                        elastic_modulus=beam_E,
                                        poisson_ratio=beam_nu,
                                        I=I)
        
        return (beam_material.I,)+tuple(
            getattr(beam_material, name)
            for name in ["E", "nu"]
        )


class ChannelBeamMaterial(CNodeType):
    r"""Timoshenko Beam Material Definition Node.

    Inputs:
        property (STRING): Material type, e.g., "Steel".
        beam_type (MENU): Beam model type selection.
        mu_y (FLOAT): Ratio of maximum to average shear stress for y-direction shear.
        mu_z (FLOAT): Ratio of maximum to average shear stress for z-direction shear.
        beam_E (FLOAT): Elastic modulus of the beam material.
        beam_nu (FLOAT): Poisson’s ratio of the beam material.

    Outputs:
        property (STRING): Material type.
        beam_type (MENU): Beam model type.
        E (FLOAT): Elastic modulus of the beam material.
        mu (FLOAT): Shear modulus, computed as `E / [2(1 + nu)]`.
    """
    TITLE: str = "槽形梁材料属性"
    PATH: str = "material.solid"
    DESC: str = """该节点用于定义槽形梁的材料属性，并根据输入的梁几何参数和材料参数计算材料的基本力学常数，
        包括弹性模量、泊松比和剪切模量。"""
        
    INPUT_SLOTS = [
        PortConf("property", DataType.STRING, 0, desc="材料名称（如钢、铝等）", title="材料材质", default="Steel"),
        PortConf("beam_type", DataType.MENU, 0, desc="轮轴材料类型选择", title="梁材料", default="Timo_beam", 
                items=["Euler_beam", "Timo_beam"]),
        PortConf("mu_y", DataType.FLOAT, 0, desc="y方向剪切应力的最大值与平均值比例因子", 
                 title="y向剪切因子", default=2.44),
        PortConf("mu_z", DataType.FLOAT, 0, desc="z方向剪切应力的最大值与平均值比例因子", 
                 title="z向剪切因子", default=2.38),
        PortConf("beam_E", DataType.FLOAT, 0, desc="梁的弹性模量", title="梁的弹性模量", default=2.1e11),
        PortConf("beam_nu", DataType.FLOAT, 0, desc="梁的泊松比", title="梁的泊松比", default=0.25),
        PortConf("beam_density", DataType.FLOAT, 0, desc="梁的密度", title="梁的密度", default=7800)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("E", DataType.FLOAT, title="梁的弹性模量"),
        PortConf("nu", DataType.FLOAT, title="梁的泊松比"),
        PortConf("mu", DataType.FLOAT, title="梁的剪切模量"),
        PortConf("rho", DataType.FLOAT, title="梁的密度")
    ]
    
    @staticmethod
    def run(property="Steel", beam_type="Timoshemko_beam", 
            mu_y=2.44, mu_z=2.38,
            beam_E=2.1e11, beam_nu=0.3, beam_density=7800):
        from fealpy.csm.model.beam.channel_beam_data_3d import ChannelBeamData3D
        from fealpy.csm.material import TimoshenkoBeamMaterial
        
        model = ChannelBeamData3D(mu_y=mu_y, mu_z=mu_z)
        
        beam_material = TimoshenkoBeamMaterial(model=model, 
                                        name=beam_type,
                                        elastic_modulus=beam_E,
                                        poisson_ratio=beam_nu,
                                        density=beam_density)

        return tuple(
            getattr(beam_material, name)
            for name in ["E", "nu", "mu", "rho"]
        )
        

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
    TITLE: str = "槽形梁应变应力计算"
    PATH: str = "material.solid"
    DESC: str = """该节点用于计算槽形梁在给定载荷和边界条件下的应变和应力分布。"""
    INPUT_SLOTS = [
        PortConf("mu_y", DataType.FLOAT, 1, desc="y方向剪切应力的最大值与平均值比例因子", 
                 title="y向剪切因子"),        
        PortConf("mu_z", DataType.FLOAT, 1, desc="z方向剪切应力的最大值与平均值比例因子", 
                 title="z向剪切因子"),
        PortConf("beam_E", DataType.FLOAT, 1, desc="梁的弹性模量", title="梁的弹性模量"),
        PortConf("beam_nu", DataType.FLOAT, 1, desc="梁的泊松比", title="梁的泊松比"),
        PortConf("mesh", DataType.MESH, 1, desc="槽形梁的三维网格", title="梁网格"),
        PortConf("uh", DataType.TENSOR, 1, desc="有限元分析得到的位移解向量", title="位移解"),
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
        
        beam_E = options.get("beam_E")
        beam_nu = options.get("beam_nu")
        material = TimoshenkoBeamMaterial(model=model, 
                                        name="Timoshenko_beam",
                                        elastic_modulus=beam_E,
                                        poisson_ratio=beam_nu)
        
        mesh = options.get("mesh")
        disp = options.get("uh")
        uh = disp.reshape(-1, 6)
        y = options.get("y", 0.0)
        z = options.get("z", 0.0)
        strain, stress = material.compute_strain_and_stress(
            mesh=mesh,
            disp=uh,
            cross_section_coords=(y, z),
            axial_position=None,
            coord_transform=model.coord_transform(),
            ele_indices=None
        )
        
        return strain, stress