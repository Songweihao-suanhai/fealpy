
from ..nodetype import CNodeType, PortConf, DataType

def lagrange_multiplier(A, b, c=0, uspace=None, pspace=None):
    """
    Constructs the augmented system matrix for Lagrange multipliers.
    c is the integral of pressure, default is 0.
    """
    from fealpy.sparse import COOTensor
    from fealpy.backend import backend_manager as bm
    from fealpy.fem import SourceIntegrator, LinearForm
    from fealpy.fem import BlockForm
    LagLinearForm = LinearForm(pspace)
    LagLinearForm.add_integrator(SourceIntegrator(source=1))
    LagA = LagLinearForm.assembly()
    LagA = bm.concatenate([bm.zeros(uspace.number_of_global_dofs()), LagA], axis=0)

    A1 = COOTensor(bm.array([bm.zeros(len(LagA), dtype=bm.int32),
                            bm.arange(len(LagA), dtype=bm.int32)]), LagA, spshape=(1, len(LagA)))

    A = BlockForm([[A, A1.T], [A1, None]])
    A = A.assembly_sparse_matrix(format='csr')
    b0 = bm.array([c])
    b  = bm.concatenate([b, b0], axis=0)
    return A, b

__all__ = ["StationaryNSRun"]

class StationaryNSRun(CNodeType):
    r"""Finite element iterative solver for steady incompressible Navier-Stokes equations.

    Inputs:
        maxstep (int): Maximum number of nonlinear iterations.
        tol (float): Convergence tolerance based on velocity and pressure residuals.
        update (function): Function to update coefficients or nonlinear terms.
        apply_bc (function): Function to apply Dirichlet boundary conditions.
        BForm (linops): Bilinear form operator for system matrix assembly.
        LForm (linops): Linear form operator for right-hand side vector assembly.
        uspace(space): Velocity function space.
        pspace(space): Pressure function space.
        mesh(mesh): Computational mesh.

    Outputs:
        uh (tensor): Final numerical velocity field.
        uh_x (tensor): x-component of the velocity field.
        uh_y (tensor): y-component of the velocity field.
        ph (tensor): Final numerical pressure field.
    """
    TITLE: str = "稳态 NS 方程有限元迭代求解"
    PATH: str = "simulation.solvers"
    DESC: str = """该节点实现稳态不可压 Navier-Stokes 方程的有限元迭代求解器，通过系数更新与边界
                条件施加，逐步组装并求解线性系统，输出速度场与压力场的稳态数值解。"""
    INPUT_SLOTS = [
        PortConf("maxstep", DataType.INT, 0, default=1000, min_val=1, title="最大迭代步数"),
        PortConf("tol", DataType.FLOAT, 0, default=1e-6, min_val=1e-12, max_val=1e-2, title="残差"),
        PortConf("update", DataType.FUNCTION, title="更新函数"),
        PortConf("apply_bc", DataType.FUNCTION, title="边界处理函数"),
        PortConf("BForm", DataType.LINOPS, title="算子"),
        PortConf("LForm", DataType.LINOPS, title="向量"),
        PortConf("uspace", DataType.SPACE, title="速度函数空间"),
        PortConf("pspace", DataType.SPACE, title="压力函数空间"),
        PortConf("mesh", DataType.MESH, title="网格")
    ]
    OUTPUT_SLOTS = [
        PortConf("uh", DataType.TENSOR, title="速度数值解"),
        PortConf("uh_x", DataType.TENSOR, title="速度x分量数值解"),
        PortConf("uh_y", DataType.TENSOR, title="速度y分量数值解"),
        PortConf("ph", DataType.TENSOR, title="压力数值解")
    ]
    @staticmethod
    def run(maxstep, tol, update, apply_bc, BForm, LForm, uspace, pspace, mesh):
        from fealpy.solver import spsolve
        uh0 = uspace.function()
        ph0 = pspace.function()
        uh1 = uspace.function()
        ph1 = pspace.function()
        ugdof = uspace.number_of_global_dofs()
        for i in range(maxstep):
            update(uh0)
            A = BForm.assembly()
            F = LForm.assembly()
            A, F = apply_bc(A, F)
            A, F = lagrange_multiplier(A, F, c = 0, uspace=uspace, pspace=pspace)
            x = spsolve(A, F,"mumps")
            uh1[:] = x[:ugdof]
            ph1[:] = x[ugdof:-1]
            res_u = mesh.error(uh0, uh1)
            res_p = mesh.error(ph0, ph1)
            
            if res_u + res_p < tol:
                break
            uh0[:] = uh1
            ph0[:] = ph1

        NN = mesh.number_of_nodes()
        uh_x = uh1[:int(ugdof/2)]
        uh_x = uh_x[:NN]
        uh_y = uh1[int(ugdof/2):]
        uh_y = uh_y[:NN]

        return uh1, uh_x, uh_y, ph1
    
class IncompressibleNSIPCSRun(CNodeType):
    r"""IPCS solver for unsteady incompressible Navier-Stokes equations.
    Inputs:
        i (int): Current time step index.
        dt (float): Time step size.
        u0 (tensor): Previous time step velocity field.
        p0 (tensor): Previous time step pressure field.
        predict_velocity (function): Function that assembles the velocity prediction system.
        correct_pressure (function): Function that assembles the pressure correction system.
        correct_velocity (function): Function that assembles the velocity correction system.
        mesh (mesh): Computational mesh.
    Outputs:
        uh (tensor): Numerical velocity field at the current time step.
        ph (tensor): Numerical pressure field at the current time step.
    """
    TITLE: str = "IPCS 计算模型"
    PATH: str = "simulation.solvers"
    DESC: str  = """该节点实现非稳态不可压 Navier-Stokes 方程的 IPCS 分步算法求解器，按时间步推进依次完成速度预测、
                压力修正与速度校正，并输出速度与压力场的时序数值结果。"""
    INPUT_SLOTS = [
        PortConf("dt", DataType.FLOAT, 0, title="时间步长"),
        PortConf("i", DataType.FLOAT, title="当前时间步"),
        PortConf("velocity_0", DataType.FUNCTION, title="初始速度"),
        PortConf("pressure_0", DataType.FUNCTION, title="初始压力"),
        PortConf("uspace", DataType.SPACE, title="速度函数空间"),
        PortConf("pspace", DataType.SPACE, title="压力函数空间"),
        PortConf("uh0", DataType.TENSOR, title="上一时间步速度"),
        PortConf("ph0", DataType.TENSOR, title="上一时间步压力"),
        PortConf("predict_velocity", DataType.FUNCTION, title="预测速度方程离散"),
        PortConf("correct_pressure", DataType.FUNCTION, title="压力修正方程离散"),
        PortConf("correct_velocity", DataType.FUNCTION, title="速度修正方程离散"),
    ]
    OUTPUT_SLOTS = [
        PortConf("uh", DataType.TENSOR, title="速度数值解"),
        PortConf("ph", DataType.TENSOR, title="压力数值解"),
    ]
    def run(dt, i, velocity_0, pressure_0, uspace, pspace, uh0, ph0, 
            predict_velocity, correct_pressure, correct_velocity):
        from fealpy.solver import cg
        from fealpy.decorator import cartesian

        if i == 0:
            u0 = uspace.interpolate(cartesian(lambda p:velocity_0(p, 0)))
            p0 = pspace.interpolate(cartesian(lambda p:pressure_0(p, 0)))
        else:
            u0 = uh0
            p0 = ph0

        uh1 = u0.space.function()
        uhs = u0.space.function()
        ph1 = p0.space.function()
        pgdof = p0.space.number_of_global_dofs()
        
        t  = dt * (i + 1)
        A0, b0 = predict_velocity(u0, p0, t = t, dt = dt)
        uhs[:] = cg(A0, b0)

        A1, b1 = correct_pressure(uhs, p0, t = t, dt = dt)
        ph1[:] = cg(A1, b1)[:pgdof]

        A2, b2 = correct_velocity(uhs, p0, ph1, t = t, dt = dt)
        uh1[:] = cg(A2, b2)

        return uh1, ph1
