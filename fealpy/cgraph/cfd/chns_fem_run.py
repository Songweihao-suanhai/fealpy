
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["CHNSFEMModel"]

class CHNSFEMModel(CNodeType):
    r"""Cahn–Hilliard–Navier–Stokes (CHNS) Equation Finite Element Solver.
    """
    TITLE: str = " CHNS 方程有限元计算模型"
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
        PortConf("equation", DataType.LIST, 1, title="方程"),
        PortConf("boundary_condition", DataType.FUNCTION, title="边界条件"),
        PortConf("is_boundary", DataType.FUNCTION, title="边界"),
        PortConf("apply_bc", DataType.FUNCTION, title="边界处理函数", default=None),
        PortConf("ns_q", DataType.INT, 0, default = 3, min_val=3, title="NS积分精度"),
        PortConf("ch_q", DataType.INT, 0, default = 5, min_val=5, title="CH积分精度"),
        PortConf("s", DataType.FLOAT, 0, title="稳定参数", default=1.0),
        PortConf("x0", DataType.LIST, title="已知值"),
    ]
    OUTPUT_SLOTS = [
        PortConf("phi0", DataType.TENSOR, title="上一时间步相场"),
        PortConf("phi1", DataType.TENSOR, title="当前时间步相场"),
        PortConf("u0", DataType.TENSOR, title="上一时间步速度"),
        PortConf("u1", DataType.TENSOR, title="当前时间步速度"),
        PortConf("p1", DataType.TENSOR, title="当前时间步压力"),
        PortConf("rho", DataType.TENSOR, title="密度"),
        PortConf("xh", DataType.LIST, title="当前时间步解")
    ]
    @staticmethod
    def run(dt, i, equation, boundary_condition, is_boundary, apply_bc, ns_q, ch_q, s, x0):
        from fealpy.backend import backend_manager as bm
        from fealpy.solver import spsolve
        from fealpy.fem import DirichletBC
        from fealpy.decorator import variantmethod

        bm.set_backend('pytorch')
        bm.set_default_device('cpu')

        equation = equation[0]
        mobility = equation["mobility"]
        interface = equation["interface"]
        free_energy = equation["free_energy"]
        time_derivative = equation["time_derivative"]
        convection = equation["convection"]
        pressure = equation["pressure"]
        viscosity = equation["viscosity"]   
        source = equation["source"]
        phi0 = x0[0]
        phi1 = x0[1]
        u0 = x0[2]
        u1 = x0[3]
        p0 = x0[4]

        class CHNSFEM:
            def __init__(self):
                self.phispace = phi0.space
                self.uspace = u0.space
                self.pspace = p0.space
                self.boundary_condition = boundary_condition
                self.is_boundary = is_boundary
                self.ns_q = ns_q
                self.ch_q = ch_q
                self.s = s

            @variantmethod("ch_fem")
            def ch_method(self):
                self.ch_method_name = "ch_fem"
                from fealpy.cgraph.cfd.fem import CahnHilliard
                return CahnHilliard.method(self.phispace, self.ch_q, self.s)

            @variantmethod("ns_bdf2")
            def ns_method(self):
                self.ns_method_name = "BDF2"
                uspace = self.uspace
                pspace = self.pspace
                boundary_condition = self.boundary_condition
                is_boundary = self.is_boundary
                q = self.ns_q
                from fealpy.cgraph.cfd.fem import IncompressibleNS
                method = IncompressibleNS.method[self.ns_method_name]
                return method(uspace, pspace, boundary_condition, is_boundary, q)

            def run(self):
                phispace = self.phispace
                uspace = self.uspace
                pspace = self.pspace
                ch_update = self.ch_method()
                ns_update = self.ns_method()
                phigdof = phispace.number_of_global_dofs()
                ugdof = uspace.number_of_global_dofs()
                pgdof = pspace.number_of_global_dofs()
                mu1 = phispace.function()
                p1 = p0
                t = (i + 1) * dt

                ch_A, ch_b = ch_update(u0, u1, phi0, phi1, dt, 
                               mobility, interface, free_energy)
                ch_x = spsolve(ch_A, ch_b, 'mumps')

                phi2 = ch_x[:phigdof]
                mu2 = ch_x[phigdof:]  
                
                # 更新NS方程参数
                rho = time_derivative(phi1)
                ctd = time_derivative(phi1)
                cc = convection(phi1)
                body_force = source(phi1)

                if apply_bc is None:
                    (is_ux_boundary, is_uy_boundary) = is_boundary
                    is_bd = uspace.is_boundary_dof((is_ux_boundary, is_uy_boundary), method='interp')
                    is_bd = bm.concatenate((is_bd, bm.zeros(pgdof, dtype=bm.bool)))
                    gd = bm.concatenate((bm.zeros(ugdof, dtype=bm.float64), bm.zeros(pgdof, dtype=bm.float64)))
                    BC = DirichletBC((uspace, pspace), gd=gd, threshold=is_bd, method='interp')
                    ns_A, ns_b = ns_update(u0, u1, dt, phi1, ctd = ctd, cc = cc,
                                        pc = pressure, cv = viscosity, 
                                        cbf = body_force, apply_bc = BC.apply)
                else:
                    ns_A, ns_b = ns_update(u0, u1, dt, phi1, ctd = ctd, cc = cc,
                                        pc = pressure, cv = viscosity, 
                                        cbf = body_force, apply_bc = apply_bc)
                    
                ns_x = spsolve(ns_A, ns_b, 'mumps')
                
                u0[:] = u1[:]
                u1[:] = ns_x[:ugdof]
                p1[:] = ns_x[ugdof:]
                    
                phi0[:] = phi1[:]
                phi1[:] = phi2[:]
                mu1[:] = mu2[:]

                return phi0, phi1, u0, u1, p1, rho
        
        model = CHNSFEM()
        phi0, phi1, u0, u1, p1, rho = model.run()
        xh = [phi0, phi1, u0, u1, p1]

        return phi0, phi1, u0, u1, p1, rho, xh