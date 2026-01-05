
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["ACNSMathmatics"]

class ACNSMathmatics(CNodeType):
    TITLE: str = "ACNS 数学模型"
    PATH: str = "examples.CFD"
    INPUT_SLOTS = [
        PortConf("phi", DataType.TENSOR, title="相场"),
        PortConf("u", DataType.TENSOR, title="速度"),
        PortConf("p", DataType.TENSOR, title="压力")
    ]
    OUTPUT_SLOTS = [
        PortConf("equation", DataType.DICT, title="方程"),
        PortConf("boundary", DataType.FUNCTION, title="边界条件"),
        PortConf("x0", DataType.DICT, title="初始值")
    ]
    @staticmethod
    def run(phi, u, p):
        from fealpy.backend import backend_manager as bm
        from fealpy.decorator import barycentric, cartesian
        mesh = phi.space.mesh
        d = mesh.d
        epsilon = mesh.epsilon
        g = 1.0

        @cartesian
        def init_phase(p):
            """
            Initial phase function.
            """
            x = p[...,0]
            y = p[...,1]
            r = bm.sqrt(x**2 + y**2)
            val = -bm.tanh((r - 0.5*d)/ (epsilon))
            return val
        
        @cartesian
        def phase_force(p, t):
            """
            Phase function source term.
            """
            x = p[...,0]
            return bm.zeros_like(x, dtype=bm.float64)
        
        @cartesian
        def init_velocity(p):
            """
            Initial velocity.
            """
            val = bm.zeros(p.shape, dtype=bm.float64)
            return val
        
        @cartesian
        def velocity_force(p, t):
            """
            Velocity source term.
            """
            val = bm.zeros(p.shape, dtype=bm.float64)
            val[...,1] = -g
            return val
        
        @cartesian
        def velocity_dirichlet_bc(p, t):
            """
            Velocity Dirichlet boundary condition.
            """
            val = bm.zeros(p.shape, dtype=bm.float64)
            return val
        
        @cartesian
        def init_pressure(p):
            """
            Initial pressure.
            """
            val = bm.zeros(p.shape[0], dtype=bm.float64)
            return val

        equation = {
            "velocity_force": velocity_force,
            "phase_force" : phase_force
        }

        boundary = {
            "velocity": velocity_dirichlet_bc
        }

        x0 = {
            "init_phase": init_phase,
            "init_velocity": init_velocity,
            "init_pressure": init_pressure
        }

        return (equation, boundary, x0)
        

