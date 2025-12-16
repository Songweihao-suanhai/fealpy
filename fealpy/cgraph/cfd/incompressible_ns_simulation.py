from ..nodetype import CNodeType, PortConf, DataType

def lagrange_multiplier(pspace, A, b, c=0):
    """
    Constructs the augmented system matrix for Lagrange multipliers.
    c is the integral of pressure, default is 0.
    """
    from fealpy.backend import backend_manager as bm
    from fealpy.sparse import COOTensor
    from fealpy.fem import LinearForm, SourceIntegrator, BlockForm
    LagLinearForm = LinearForm(pspace)
    LagLinearForm.add_integrator(SourceIntegrator(source=1))
    LagA = LagLinearForm.assembly()

    A1 = COOTensor(bm.array([bm.zeros(len(LagA), dtype=bm.int32),
                            bm.arange(len(LagA), dtype=bm.int32)]), LagA, spshape=(1, len(LagA)))

    A = BlockForm([[A, A1.T], [A1, None]])
    A = A.assembly_sparse_matrix(format='csr')
    b0 = bm.array([c])
    b  = bm.concatenate([b, b0], axis=0)
    return A, b

def apply_bc(space, dirichlet, is_boundary, t):
    r"""Dirichlet boundary conditions for unsteady Navier-Stokes equations."""
    from fealpy.fem import DirichletBC
    from fealpy.decorator import cartesian
    BC = DirichletBC(space=space,
            gd = cartesian(lambda p : dirichlet(p, t)),
            threshold = is_boundary,
            method = 'interp')
    return BC.apply
    

class IncompressibleNSBDF2(CNodeType):
    r"""Unsteady Incompressible Navier-Stokes solver using the BDF2 algorithm.

    Inputs:
        Re (float): Reynolds number.
        uspace(space): Function space for the velocity field.
        pspace(space): Function space for the pressure field.
        q (int): Quadrature order for numerical integration (default: 3).   
    Outputs:
        update (function): Function that assembles the system for each time step.
    """
    TITLE: str = "非稳态 NS 方程 BDF2 算法"
    PATH: str = "simulation.discretization"
    DESC: str = """
                该节点基于有限元法实现不可压 Navier-Stokes 方程的非稳态求解，采用 BDF2（二阶
                向后差分格式）进行时间离散。在每个时间步中，该算法构建速度与压力场的耦合离散系统，
                通过定义双线性型与线性型实现刚度矩阵与载荷项的组装，同时可支持不同雷诺数、积分精
                度与源项输入。输出为一个时间步进函数，用于每步更新系统矩阵 A 与右端项 L。
                
                使用示例：用户可在输入槽中传入速度与压力的有限元空间 (uspace, pspace)、雷诺数
                (Re) 并设置积分精度 (q)，输出的 update 函数可被上层时间推进框架调用，以在每个
                时间步组装系统方程。
                """
    INPUT_SLOTS = [
        PortConf("u", DataType.TENSOR, 1, title="速度"),
        PortConf("p", DataType.TENSOR, 1, title="压力"),
        PortConf("dirichlet_boundary", DataType.FUNCTION, title="边界条件"),
        PortConf("is_boundary", DataType.FUNCTION, title="边界"),
        PortConf("q", DataType.INT, 0, default = 3, min_val=3, title="积分精度")
    ]
    OUTPUT_SLOTS = [
        PortConf("update", DataType.FUNCTION, title="BDF2 离散格式")
    ]

    @staticmethod
    def run(u, p, dirichlet_boundary, is_boundary, q):
        from fealpy.fem import (BilinearForm, BlockForm, ScalarMassIntegrator, 
                                ScalarConvectionIntegrator, ViscousWorkIntegrator,
                                PressWorkIntegrator, LinearBlockForm, LinearForm, 
                                SourceIntegrator)
        from fealpy.backend import backend_manager as bm
        from fealpy.decorator import barycentric, cartesian
        uspace = u.space
        pspace = p.space

        def update(u_0, u_1, dt, t, ctd, cc, pc, cv, cbf, apply_bc = True):
            
            ## BilinearForm
            
            A00 = BilinearForm(uspace)
            BM = ScalarMassIntegrator(q=q)
            BM.coef = 3*ctd/(2*dt)
            BC = ScalarConvectionIntegrator(q=q)
            def BC_coef(bcs, index): 
                ccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
                result = 2* ccoef * u_1(bcs, index)
                return result
            BC.coef = BC_coef
            BD = ViscousWorkIntegrator(q=q)
            BD.coef = 2*cv 

            A00.add_integrator(BM)
            A00.add_integrator(BC)
            A00.add_integrator(BD)

            A01 = BilinearForm((pspace, uspace))
            BPW0 = PressWorkIntegrator(q=q)
            BPW0.coef = -pc
            A01.add_integrator(BPW0) 

            A10 = BilinearForm((pspace, uspace))
            BPW1 = PressWorkIntegrator(q=q)
            BPW1.coef = -1
            A10.add_integrator(BPW1)
            
            A = BlockForm([[A00, A01], [A10.T, None]])

            ## LinearForm
            L0 = LinearForm(uspace) 
            LSI_U = SourceIntegrator(q=q)
            @barycentric
            def LSI_U_coef(bcs, index):
                masscoef = ctd(bcs, index)[..., bm.newaxis] if callable(ctd) else ctd
                result0 =  masscoef * (4*u_1(bcs, index) - u_0(bcs, index)) / (2*dt)
                
                ccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
                result1 = ccoef*bm.einsum('cqij, cqj->cqi', u_1.grad_value(bcs, index), u_0(bcs, index))
                cbfcoef = cbf(bcs, index) if callable(cbf) else cbf
                
                result = result0 + result1 + cbfcoef
                return result
            LSI_U.source = LSI_U_coef
            L0.add_integrator(LSI_U)

            L1 = LinearForm(pspace)
            L = LinearBlockForm([L0, L1])

            A = A.assembly()
            L = L.assembly()
            if apply_bc is True:
                (velocity_dirichlet, pressure_dirichlet) = dirichlet_boundary()
                (is_velocity_boundary, is_pressure_boundary) = is_boundary()
                gd_v = cartesian(lambda p: velocity_dirichlet(p, t))
                gd_p = cartesian(lambda p: pressure_dirichlet(p, t))
                gd = (gd_v, gd_p)
                is_bd = (is_velocity_boundary, is_pressure_boundary)
                apply = apply_bc((uspace, pspace), gd, is_bd, t)
                A, L = apply(A, L)
            return A, L
        
        return update


        