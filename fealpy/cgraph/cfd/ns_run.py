
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

__all__ = ["StationaryNSRun", "IncompressibleNSFEMModel"]

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
    
class IncompressibleNSFEMModel(CNodeType):
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
    TITLE: str = "不可压缩 NS 计算模型"
    PATH: str = "simulation.solvers"
    DESC: str  = """该节点实现非稳态不可压 Navier-Stokes 方程的 IPCS 分步算法求解器，按时间步推进依次完成速度预测、
                压力修正与速度校正，并输出速度与压力场的时序数值结果。"""
    INPUT_SLOTS = [
        PortConf("i", DataType.FLOAT, title="当前时间步"),
        PortConf("dt", DataType.FLOAT, 0, title="时间步长"),
        PortConf("method_name", DataType.MENU, 0, title="算法", default="IPCS", items=["IPCS", "Newton"]),
        PortConf("time_derivative", DataType.FLOAT, title="时间项系数"),
        PortConf("convection", DataType.FLOAT, title="对流项系数"),
        PortConf("pressure", DataType.FLOAT, title="压力项系数"),
        PortConf("viscosity", DataType.FLOAT, title="粘性项系数"),
        PortConf("source", DataType.FUNCTION, title="源项"),
        PortConf("dirichlet_boundary", DataType.FUNCTION, title="边界条件"),
        PortConf("is_boundary", DataType.FUNCTION, title="边界"),
        PortConf("apply_bc", DataType.FUNCTION, title="边界处理函数"),
        PortConf("q", DataType.INT, 0, default = 3, min_val=3, title="积分精度"),
        PortConf("uh0", DataType.TENSOR, title="上一时间步速度"),
        PortConf("ph0", DataType.TENSOR, title="上一时间步压力")
    ]
    OUTPUT_SLOTS = [
        PortConf("uh", DataType.TENSOR, title="速度数值解"),
        PortConf("ph", DataType.TENSOR, title="压力数值解"),
    ]
    def run(i, dt, method_name, time_derivative, convection, pressure, viscosity, 
            source, dirichlet_boundary, is_boundary, apply_bc, q, uh0, ph0):
        from fealpy.solver import cg
        from fealpy.decorator import cartesian, variantmethod, barycentric
        from fealpy.backend import backend_manager as bm
        from fealpy.fem import (BilinearForm, ScalarMassIntegrator, ScalarDiffusionIntegrator,
                                FluidBoundaryFrictionIntegrator, DirichletBC)

        class IncompressibleNSFEM:
            def __init__(self):
                self.uspace = uh0.space
                self.pspace = ph0.space
                self.q = q
                self.dirichlet_boundary = dirichlet_boundary
                self.is_boundary = is_boundary

            @variantmethod("IPCS")
            def method(self):
                self.method_name = "IPCS"
                uspace = self.uspace
                pspace = self.pspace
                dirichlet_boundary = self.dirichlet_boundary
                is_boundary = self.is_boundary
                q = self.q
                from fealpy.cgraph.cfd.fem import fem_ipcs
                return fem_ipcs(uspace, pspace, dirichlet_boundary, is_boundary, q, apply_bc=apply_bc)

            def run(self):
                if self.method_name == "IPCS":
                    predict_velocity, correct_pressure, correct_velocity = self.method()
                    uh1 = uh0.space.function()
                    uhs = uh0.space.function()
                    ph1 = ph0.space.function()
                    pgdof = ph0.space.number_of_global_dofs()
                    
                    t  = dt * (i + 1)
                    body_force = cartesian(lambda p:source(p, t))
                    A0, b0 = predict_velocity(uh0, ph0, 
                                            t = t, 
                                            dt = dt, 
                                            ctd = time_derivative, 
                                            cc = convection, 
                                            pc = pressure, 
                                            cv = viscosity, 
                                            cbf = body_force)
                    uhs[:] = cg(A0, b0)

                    A1, b1 = correct_pressure(uhs, ph0, 
                                            t = t, 
                                            dt = dt, 
                                            ctd = time_derivative, 
                                            pc = pressure)
                    ph1[:] = cg(A1, b1)[:pgdof]

                    A2, b2 = correct_velocity(uhs, ph0, ph1, t = t, dt = dt, ctd = time_derivative)
                    uh1[:] = cg(A2, b2)

                    return uh1, ph1
                
                else:
                    pass
        
        model = IncompressibleNSFEM()
        model.method[method_name]()
        uh, ph = model.run()
        return uh, ph
                    