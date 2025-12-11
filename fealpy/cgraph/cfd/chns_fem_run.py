
from ..nodetype import CNodeType, PortConf, DataType

class CHNSFEMRun(CNodeType):
    r"""Finite Element Solver for the Coupled Cahn–Hilliard–Navier–Stokes (CHNS) Equations.

    This node implements a time-dependent finite element solver for the two-phase incompressible flow
    described by the **Cahn–Hilliard–Navier–Stokes (CHNS)** equations. The CHNS model captures 
    interface dynamics between immiscible fluids using a phase-field formulation coupled with
    Navier–Stokes equations.

    The solver advances the system in time using a given time step (`dt`) and total number of steps (`nt`).
    It alternates between solving:
        - **Cahn–Hilliard (CH)** equation: updates the phase field (`phi`) and chemical potential (`mu`);
        - **Navier–Stokes (NS)** equation: updates the velocity (`u`) and pressure (`p`) fields,
          with the local density `rho(phi)` determined by the phase field.

    Inputs:
        dt (float): Time step size.
        nt (int): Number of time steps.
        rho_up (float): Density of the upper fluid.
        rho_down (float): Density of the lower fluid.
        Fr (float): Froude number (controls gravitational effects).
        ns_update (function): Function that assembles the Navier–Stokes system.
        ch_update (function): Function that assembles the Cahn–Hilliard system.
        phispace (SpaceType): Finite element function space for the phase field.
        uspace (SpaceType): Function space for the velocity field.
        pspace (SpaceType): Function space for the pressure field.
        is_ux_boundary (function): Predicate for x-velocity component boundary.
        is_uy_boundary (function): Predicate for y-velocity component boundary.
        init_interface (function): Initial phase-field (interface) function.
        mesh (MeshType): Finite element mesh.

    Outputs:
        u (Function): Velocity vector field at the final time.
        ux (Function): x-component of velocity field.
        uy (Function): y-component of velocity field.
        p (Function): Pressure field.
        phi (Function): Phase-field function.
    """
    TITLE: str = "有限元求解 CHNS 方程"
    PATH: str = "simulation.solvers"
    DESC: str = """该节点实现了两相不可压流体的 Cahn–Hilliard–Navier–Stokes (CHNS)simulation.discretization"
                器。CHNS 模型结合了相场法与流体力学方程，用以描述两种不可混溶流体的界面演化与流动耦合过程。
                通过时间步推进（dt, nt），程序在每个时间步内依次执行：
                1. Cahn–Hilliard 方程：更新相场函数 φ 与化学势 μ；
                2. Navier–Stokes 方程：根据当前相场计算密度场 ρ(φ)，更新速度场 u 与压力场 p。

                输入参数：
                - dt ：时间步长；
                - nt ：总时间步数；
                - rho_up 、rho_down ：上下两种流体的密度；
                - Fr ：弗劳德数，用于控制重力源项；
                - ns_update ：组装 NS 方程离散系统的函数；
                - ch_update ：组装 CH 方程离散系统的函数；
                - phispace、uspace、pspace ：分别为相场、速度与压力的有限元空间；
                - is_ux_boundary、is_uy_boundary ：定义速度边界条件的判定函数；
                - init_interface ：初始界面函数；
                - mesh ：计算区域网格。

                输出结果：
                - u ：最终时刻的速度场；
                - ux、uy ：速度在 x、y 方向的分量；
                - p ：压力场；
                - phi ：相场函数。

                使用示例：
                可将“相场更新模块 (CH)”与“流体更新模块 (NS)”分别连接到 `ch_update` 与 `ns_update`，
                设置好初始界面与网格信息后，即可在多时间步迭代中自动完成流体界面的演化与速度场的时序求解。
                """
    INPUT_SLOTS = [
        PortConf("dt", DataType.FLOAT, 0, title="时间步长"),
        PortConf("i", DataType.INT, title="当前时间步"),
        PortConf("mobility", DataType.FLOAT, title="迁移率"),
        PortConf("interface", DataType.FLOAT, title="界面参数"),
        PortConf("free_energy", DataType.FLOAT, title="自由能参数"),
        PortConf("time_derivative", DataType.FUNCTION, title="时间项系数"),
        PortConf("convection", DataType.FUNCTION, title="对流项系数"),
        PortConf("pressure", DataType.FUNCTION, title="压力项系数"),
        PortConf("viscosity", DataType.FUNCTION, title="粘性项系数"),
        PortConf("source", DataType.FUNCTION, title="源项"),
        PortConf("phi0", DataType.TENSOR, title="上上时间步相场"),
        PortConf("phi1", DataType.TENSOR, title="上一时间步相场"),
        PortConf("u0", DataType.TENSOR, title="上上时间步速度"),
        PortConf("u1", DataType.TENSOR, title="上一时间步速度"),
        PortConf("p0", DataType.TENSOR, title="上一时间步压力"),
        PortConf("ns_update", DataType.FUNCTION, title="NS 更新函数"),
        PortConf("ch_update", DataType.FUNCTION, title="CH 更新函数"),
        PortConf("is_boundary", DataType.FUNCTION, title="边界"),
    ]
    OUTPUT_SLOTS = [
        PortConf("u", DataType.FUNCTION, title="速度"),
        PortConf("p", DataType.FUNCTION, title="压力"),
        PortConf("phi", DataType.FUNCTION, title="相场函数"),
        PortConf("rho", DataType.FUNCTION, title="密度")
    ]
    @staticmethod
    def run(dt, i, mobility, interface, free_energy, time_derivative, convection, 
            pressure, viscosity, source, phi0, phi1, u0, u1, p0,
            ns_update, ch_update, is_boundary):
        from fealpy.backend import backend_manager as bm
        from fealpy.solver import spsolve
        from fealpy.fem import DirichletBC

        bm.set_backend('pytorch')
        bm.set_default_device('cpu')

        phispace = phi0.space
        uspace = u0.space
        pspace = p0.space
        p1 = p0
        phigdof = phispace.number_of_global_dofs()
        ugdof = uspace.number_of_global_dofs()
        pgdof = pspace.number_of_global_dofs()
        mu1 = phispace.function()
        t = (i + 1) * dt

        # node = mesh.entity_barycenter('node')
        # tol = 1e-14
        # left_bd = bm.where(bm.abs(node[:, 0]) < tol)[0]
        # right_bd = bm.where(bm.abs(node[:, 0]-1.0) < tol)[0]
        
        ch_A, ch_b = ch_update(u0, u1, phi0, phi1, dt, 
                               mobility, interface, free_energy)
        ch_A = ch_A.assembly()
        ch_b = ch_b.assembly()
        ch_x = spsolve(ch_A, ch_b, 'mumps')

        phi2 = ch_x[:phigdof]
        mu2 = ch_x[phigdof:]  
        
        # 更新NS方程参数
        rho = time_derivative(phi1)
        ctd = time_derivative(phi1)
        cc = convection(phi1)
        body_force = source(phi1)
        ns_A, ns_b = ns_update(u0, u1, dt, phi1, ctd = ctd, cc = cc,
                               pc = pressure, cv = viscosity, 
                               cbf = body_force, apply_bc = False)
        (is_ux_boundary, is_uy_boundary) = is_boundary
        is_bd = uspace.is_boundary_dof((is_ux_boundary, is_uy_boundary), method='interp')
        is_bd = bm.concatenate((is_bd, bm.zeros(pgdof, dtype=bm.bool)))
        gd = bm.concatenate((bm.zeros(ugdof, dtype=bm.float64), bm.zeros(pgdof, dtype=bm.float64)))
        BC = DirichletBC((uspace, pspace), gd=gd, threshold=is_bd, method='interp')
        ns_A, ns_b = BC.apply(ns_A, ns_b)
        ns_x = spsolve(ns_A, ns_b, 'mumps')
        
        u0[:] = u1[:]
        u1[:] = ns_x[:ugdof]
        p1[:] = ns_x[ugdof:]
            
        phi0[:] = phi1[:]
        phi1[:] = phi2[:]
        mu1[:] = mu2[:]

        # phi2_lbdval = phi2[left_bd]
        # mask = bm.abs(phi2_lbdval) < 0.5
        # index = left_bd[mask]
        # left_point = node[index, :]
        # print("界面与左边界交点:", left_point)

        # phi2_rbdval = phi2[right_bd]
        # mask = bm.abs(phi2_rbdval) < 0.5
        # index = right_bd[mask]
        # right_point = node[index, :]
        # print("界面与右边界交点:", right_point)

        return u1, p1, phi2, rho