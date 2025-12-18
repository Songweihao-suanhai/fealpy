
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["GNBC"]

class GNBC(CNodeType):
    r"""
    广义Navier边界条件(Generalized Navier boundary condition)
    """
    TITLE: str = "GNBC边界条件处理"
    PATH: str = "simulation.discretization"
    DESC: str = """
    
"""
    INPUT_SLOTS = [
        PortConf("Dirichlet", DataType.FUNCTION, 1, title="判断是否为速度Dirichlet边界"),
        PortConf("space", DataType.SPACE, 1, title="函数空间"),
        PortConf("pspace", DataType.SPACE, 1, title="压力函数空间"),
        PortConf("uspace", DataType.SPACE, 1, title="速度函数空间"),
    ]
    OUTPUT_SLOTS = [
        PortConf("apply_bc", DataType.FUNCTION, title="边界处理函数")
    ]
    
    @staticmethod
    def run(Dirichlet,space,pspace,uspace):
        from fealpy.backend import backend_manager as bm
        from fealpy.fem import DirichletBC
        ugdof = uspace.number_of_global_dofs()
        pgdof = pspace.number_of_global_dofs()
        is_uy_bd = space.is_boundary_dof(Dirichlet)
        ux_gdof = space.number_of_global_dofs()
        is_bd = bm.concatenate((bm.zeros(ux_gdof, dtype=bool), is_uy_bd, bm.zeros(pgdof, dtype=bool)))
        NS_BC = DirichletBC(space=(uspace,pspace), \
                gd=bm.zeros(ugdof+pgdof, dtype=bm.float64), \
                threshold=is_bd, method='interp')
        def apply_bc(A, b):
            return NS_BC.apply(A, b)
        
        return apply_bc