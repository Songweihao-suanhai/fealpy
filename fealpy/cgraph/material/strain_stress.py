from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["BarStrainStress",
           "TimoshenkoBeamStrainStress",
           "TimoAxleStrainStress"]


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
    TITLE: str = "三维杆单元应变-应力计算"
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


class TimoshenkoBeamStrainStress(CNodeType):
    r"""Compute Strain and Stress for Timoshenko Beam Elements.

    Inputs:
        mesh (MESH): Mesh containing node and cell information.
        uh (TENSOR): Post-processed displacement vector.
        coord_transform (TENSOR): Coordinate transformation matrix.
        y (FLOAT): Local y-coordinate in the beam cross-section for evaluation.
        z (FLOAT): Local z-coordinate in the beam cross-section for evaluation.
        axial_position (FLOAT): Evaluation position along the beam axis ∈ [0, L].
            If None, the value is evaluated at the element midpoint L/2.

    Outputs:
        strain (TENSOR): Strain of the Timoshenko beam elements.
        stress (TENSOR): Stress of the Timoshenko beam elements.
    """
    TITLE: str = "三维铁木辛柯梁应变-应力计算"
    PATH: str = "material.solid"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, desc="包含节点和单元信息的网格", title="网格"),
        PortConf("uh", DataType.TENSOR, 1, desc="未经后处理的位移向量", title="全局位移"),
        PortConf("coord_transform", DataType.TENSOR, 1, desc="坐标变换矩阵", title="坐标变换"),
        PortConf("y", DataType.FLOAT, 0, desc="截面局部 Y 坐标", title="Y坐标", default=0.0),
        PortConf("z", DataType.FLOAT, 0, desc="截面局部 Z 坐标", title="Z坐标", default=0.0),
        PortConf("axial_position", DataType.FLOAT, 0, desc="轴向评估位置 [0, L]", title="轴向位置", default=None),
    ]
    
    OUTPUT_SLOTS = [
        PortConf("strain", DataType.TENSOR, title="应变"),
        PortConf("stress", DataType.TENSOR, title="应力")
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.csm.material import TimoshenkoBeamMaterial
        
        mesh = options.get("mesh")
        
        E_scalar = float(mesh.celldata['E'][0])
        nu_scalar = float(mesh.celldata['nu'][0])
        
        class SimpleBeamModel:
            def __init__(self):
                # 截面属性
                self.Ay = mesh.celldata['Ay'] if 'Ay' in mesh.celldata else [1.0] * mesh.number_of_cells()
                self.Az = mesh.celldata['Az'] if 'Az' in mesh.celldata else [1.0] * mesh.number_of_cells()
                self.Iy = mesh.celldata['Iy'] if 'Iy' in mesh.celldata else [1.0] * mesh.number_of_cells()
                self.Iz = mesh.celldata['Iz'] if 'Iz' in mesh.celldata else [1.0] * mesh.number_of_cells()
                
                # 剪切修正系数
                if 'mu_y' in mesh.celldata and 'mu_z' in mesh.celldata:
                    self.FSY = float(mesh.celldata['mu_y'][0])
                    self.FSZ = float(mesh.celldata['mu_z'][0])
                elif 'shear_factor' in mesh.celldata:
                    # 如果有 shear_factor,使用它作为剪切修正系数
                    self.FSY = float(mesh.celldata['shear_factor'][0])
                    self.FSZ = float(mesh.celldata['shear_factor'][0])
                    
        model = SimpleBeamModel()
        
        material = TimoshenkoBeamMaterial(
            model=model,
            name="Timoshenko_beam",
            elastic_modulus=E_scalar,
            poisson_ratio=nu_scalar
        )
        
        uh = options.get("uh").reshape(-1, 6)
        
        # 获取截面坐标和轴向位置
        y = options.get("y")
        z = options.get("z")
        axial_position = options.get("axial_position")
        
        # 计算应变和应力
        strain, stress = material.compute_strain_and_stress(
            mesh=mesh,
            disp=uh,
            cross_section_coords=(y, z),
            axial_position=axial_position,
            coord_transform=options.get("coord_transform"),
            ele_indices=None
        )
        
        return strain, stress


class TimoAxleStrainStress(CNodeType):
    r"""Compute Strain and Stress for Beam-Axle Coupled Elements.
    
        Inputs:
            mesh (MESH): Mesh containing node, cell information and celldata['type'].
                     celldata['type']: 0=beam elements, 1=axle elements.
            uh (TENSOR): Post-processed displacement vector.
            y (FLOAT): Local y-coordinate in the beam cross-section for evaluation.
            z (FLOAT): Local z-coordinate in the beam cross-section for evaluation.
            axial_position (FLOAT): Evaluation position along the beam axis ∈ [0, L].
                If None, the value is evaluated at the element midpoint L/2.
            R_beam (TENSOR): Coordinate transformation matrix for beam elements.
            R_axle (TENSOR): Coordinate transformation matrix for axle elements.

        Outputs:
            strain (TENSOR): Combined strain tensor for all elements (NC, 3).
            stress (TENSOR): Combined stress tensor for all elements (NC, 3).
    """
    TITLE: str = "梁-轴耦合单元应变-应力计算"
    PATH: str = "material.solid"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, desc="包含节点和单元信息的网格", title="网格"),
        PortConf("uh", DataType.TENSOR, 1, desc="未经后处理的位移向量", title="位移向量"),
        PortConf("y", DataType.FLOAT, 0, desc="截面局部 Y 坐标", title="Y坐标", default=0.0),
        PortConf("z", DataType.FLOAT, 0, desc="截面局部 Z 坐标", title="Z坐标", default=0.0),
        PortConf("axial_position", DataType.FLOAT, 0, desc="轴向评估位置", title="轴向位置", default=None),
        PortConf("R_beam", DataType.TENSOR, 1, desc="列车轮轴梁单元部分坐标变换矩阵", title="梁单元坐标变换"),
        PortConf("R_axle", DataType.TENSOR, 1, desc="列车轮轴轴单元部分坐标变换矩阵", title="轴单元坐标变换")
    ]
    
    OUTPUT_SLOTS = [
        PortConf("strain", DataType.TENSOR, title="应变"),
        PortConf("stress", DataType.TENSOR, title="应力")
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.backend import backend_manager as bm
        from fealpy.csm.material import TimoshenkoBeamMaterial
        from fealpy.csm.material import AxleMaterial
        
        mesh = options.get("mesh")
        NC = mesh.number_of_cells()
        
        cell_types = mesh.celldata['type']
        beam_indices = bm.where(cell_types == 0)[0]
        axle_indices = bm.where(cell_types == 1)[0]
        
        class SimpleBeamModel:
            def __init__(self):
                self.GD = 3
                self.Ax = mesh.celldata['Ax'][beam_indices]
                self.Ay = mesh.celldata['Ay'][beam_indices]
                self.Az = mesh.celldata['Az'][beam_indices]
                self.Iy = mesh.celldata['Iy'][beam_indices]
                self.Iz = mesh.celldata['Iz'][beam_indices]
                self.J = mesh.celldata['J'][beam_indices]
                
                shear_factor_data = mesh.celldata['shear_factor'][beam_indices]
                self.FSY = float(shear_factor_data[0]) 
                self.FSZ = float(shear_factor_data[0]) 
                    
        model = SimpleBeamModel()
        
        E_array = mesh.celldata['E'][beam_indices]
        nu_array = mesh.celldata['nu'][beam_indices]
        E = float(E_array[0])
        nu = float(nu_array[0])

        beam_material = TimoshenkoBeamMaterial(model=model,
                                        name="Timo_beam",
                                        elastic_modulus=E,
                                        poisson_ratio=nu)

        k_axle_array = mesh.celldata['k_axle'][axle_indices]
        k_axle = float(k_axle_array[0])
        axle_material = AxleMaterial(
            model=model,
            name="bar",
            k_axle=k_axle,
            elastic_modulus=k_axle
        )
          
        uh = options.get("uh").reshape(-1, 2*model.GD)
        y = options.get("y")
        z = options.get("z")
        
        axial_position = options.get("axial_position")
        R_beam = options.get("R_beam")
        R_axle = options.get("R_axle")
        
        beam_strain, beam_stress = beam_material.compute_strain_and_stress(
                        mesh=mesh,
                        disp=uh,
                        cross_section_coords=(y, z),
                        axial_position=axial_position,
                        coord_transform=R_beam,
                        ele_indices=beam_indices)
        
        axle_strain, axle_stress = axle_material.compute_strain_and_stress(
                        mesh=mesh,
                        disp=uh,
                        coord_transform=R_axle,
                        ele_indices=axle_indices)

        strain = bm.zeros((NC, 3), dtype=bm.float64)
        stress = bm.zeros((NC, 3), dtype=bm.float64)
        
        strain[beam_indices] = beam_strain
        stress[beam_indices] = beam_stress
        
        strain[axle_indices] = axle_strain
        stress[axle_indices] = axle_stress
        
        return strain, stress