from fealpy.decorator import  cartesian
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["GNBC"]

class GNBC(CNodeType):
    r"""
    广义Navier边界条件(Generalized Navier boundary condition)处理
    """
    TITLE: str = "GNBC边界条件处理"
    PATH: str = "simulation.discretization"
    DESC: str = """
    
"""
    INPUT_SLOTS = [
        PortConf("equation", DataType.LIST, title="方程"),
        PortConf("u", DataType.TENSOR, title="速度"),
        PortConf("p", DataType.TENSOR, title="压力"),
        PortConf("dt", DataType.FLOAT, title="时间步长"),
        PortConf("is_wall_boundary", DataType.FUNCTION, 1, title="判断是否为壁面边界"),
        PortConf("q", DataType.INT, 0, title="积分次数", default=5)
    ]
    OUTPUT_SLOTS = [
        PortConf("apply_bc", DataType.FUNCTION, title="边界处理函数"),
    ]
    
    @staticmethod
    def run(equation,u,p,dt,is_wall_boundary,q=5):
        from fealpy.backend import backend_manager as bm
        from fealpy.fem import DirichletBC,BilinearForm,LinearForm,BlockForm,LinearBlockForm
        from fealpy.fem import BoundaryFaceSourceIntegrator,TangentFaceMassIntegrator
        equation = equation[0]
        Dirichlet = equation["is_uy_Dirichlet"]
        L_s = equation["L_s"]
        u_w = equation["u_w"]
        def apply_bc(A0, b0):
            uspace = u.space
            pspace = p.space
            A00 = BilinearForm(uspace)
            A01 = BilinearForm((pspace, uspace))
            A10 = BilinearForm((pspace, uspace))
            FM = TangentFaceMassIntegrator(coef=2*dt/L_s, q=q, threshold=is_wall_boundary)
            A00.add_integrator(FM)
            A = BlockForm([[A00, A01], [A10.T, None]]).assembly() 
            
            L0 = LinearForm(uspace) 
            @cartesian
            def uw_BF_SI_coef(p):
                return (2*dt/L_s)*u_w(p)
            uw_BF_SI = BoundaryFaceSourceIntegrator(source=uw_BF_SI_coef, q=q, threshold=is_wall_boundary)
            L0.add_integrator(uw_BF_SI)
            L1 = LinearForm(pspace)
            L = LinearBlockForm([L0, L1]).assembly()
            ugdof = uspace.number_of_global_dofs()
            pgdof = pspace.number_of_global_dofs()
            space = uspace.scalar_space
            is_uy_bd = space.is_boundary_dof(Dirichlet)
            ux_gdof = space.number_of_global_dofs()
            is_bd = bm.concatenate((bm.zeros(ux_gdof, dtype=bool), is_uy_bd, bm.zeros(pgdof, dtype=bool)))
            NS_BC = DirichletBC(space=(uspace,pspace), \
                    gd=bm.zeros(ugdof+pgdof, dtype=bm.float64), \
                    threshold=is_bd, method='interp')
            A,L = NS_BC.apply(A, L)
            A = A + A0
            L = L + b0
            return A,L
        
        return apply_bc