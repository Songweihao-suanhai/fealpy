from ..nodetype import CNodeType, PortConf, DataType

__all__ = ['SpringStiffnessAssembly']


class SpringStiffnessAssembly(CNodeType):
    r"""
    
    """
    TITLE: str = "弹簧单元刚度矩阵组装"
    PATH: str = "simulation.discretization" 
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, 
                desc="包含弹簧单元的网格对象", 
                title="网格"),
        PortConf("R", DataType.TENSOR, 1,
                desc="坐标变换矩阵 (NC_spring, 12, 12)",
                title="坐标变换矩阵"),
        PortConf("rotation_factor", DataType.FLOAT, 0,
                desc="转动刚度放大系数",
                title="转动刚度系数",
                default=1e3)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("K", DataType.TENSOR, title="全局刚度矩阵")
    ]
    
    @staticmethod
    def run(**options):
        from fealpy.backend import bm
        from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
        
        mesh = options.get("mesh")
        R = options.get("R")
        rotation_factor = options.get("rotation_factor")
        
        space = LagrangeFESpace(mesh, p=1)
        tspace = TensorFunctionSpace(space, shape=(-1, 6))
        
        cell = mesh.entity('cell')
        cell_types = mesh.celldata.get('type', bm.zeros(len(cell), dtype=bm.int32))
        is_spring = cell_types == 1
        spring_indices = bm.where(is_spring)[0]
        
        if len(spring_indices) == 0:
            Dofs = tspace.number_of_global_dofs()
            K = bm.zeros((Dofs, Dofs), dtype=bm.float64)
            return K
        
        k_spring = mesh.celldata['k_spring'][spring_indices]
        
        K0 = bm.zeros((len(spring_indices), 3, 3), dtype=bm.float64)
        K0[:, 0, 0] = k_spring  # kx
        K0[:, 1, 1] = k_spring  # ky
        K0[:, 2, 2] = k_spring  # kz
        
        kr = k_spring * rotation_factor
        one = bm.ones((3, 3), dtype=bm.float64)
        K_rot = bm.einsum('c, ij -> cij', kr, one)
        
        row1 = bm.concatenate([K0, K_rot, -K0, K_rot], axis=2)
        row2 = bm.concatenate([K_rot, K_rot, K_rot, K_rot], axis=2)
        row3 = bm.concatenate([-K0, K_rot, K0, K_rot], axis=2)
        row4 = bm.concatenate([K_rot, K_rot, K_rot, K_rot], axis=2)
        
        KE_local = bm.concatenate([row1, row2, row3, row4], axis=1)
        KE = bm.einsum('cji, cjk, ckl -> cil', R, KE_local, R) # (NC_spring, 12, 12)
        
        cell2dof = tspace.cell_to_dof()[spring_indices]
        Dofs = tspace.number_of_global_dofs()
        K = bm.zeros((Dofs, Dofs), dtype=bm.float64)
        
        for i, dof in enumerate(cell2dof):
            K[dof[:, None], dof] += KE[i]
        
        return K
        
        
