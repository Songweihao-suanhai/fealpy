
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
        PortConf("equation", DataType.LIST, title="方程"),
        PortConf("boundary", DataType.FUNCTION, title="边界条件"),
        PortConf("is_boundary", DataType.FUNCTION, title="边界"),
        PortConf("x0", DataType.LIST, title="初始值")
    ]
    @staticmethod
    def run(phi, u, p):
        from fealpy.backend import backend_manager as bm
        from fealpy.decorator import barycentric, cartesian
        mesh = phi.space.mesh
        rho0 = mesh.rho0
        rho1 = mesh.rho1
        mu0 = mesh.mu0
        mu1 = mesh.mu1
        lam = mesh.lam
        gamma = mesh.gamma
        epsilon = mesh.epsilon
        g = 9.8
        
        #无量纲化
        d = 0.005
        area = 2*d * 4*d
        ref_length = d
        ref_velocity = (g*d)**0.5
        ref_rho = min(rho0,rho1)
        ref_mu = ref_rho*ref_length*ref_velocity
        
        area /= ref_length**2
        # box = [x / ref_length for x in box]
        epsilon /= ref_length
        
        rho0 /= ref_rho
        rho1 /= ref_rho
        mu0 /= ref_mu
        mu1 /= ref_mu
        
        d /= ref_length
        g = 1.0


        def rho(phi):
            tag0 = phi[:] >1
            tag1 = phi[:] < -1
            phi[tag0] = 1
            phi[tag1] = -1
            rho = phi.space.function()
            rho[:] = 0.5 * (rho0 + rho1) + 0.5 * (rho0 - rho1) * phi
            return rho
        
        def mu(phi):
            tag0 = phi[:] >1
            tag1 = phi[:] < -1
            phi[tag0] = 1
            phi[tag1] = -1
            mu = phi.space.function()
            mu[:] = 0.5 * (mu0 + mu1) + 0.5 * (mu0 - mu1) * phi
            return mu

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
        def boundary(p, t):
            """
            Velocity Dirichlet boundary condition.
            """
            val = bm.zeros(p.shape, dtype=bm.float64)
            return val
        
        @cartesian
        def is_boundary(p):
            return None
        
        @cartesian
        def init_pressure(p):
            """
            Initial pressure.
            """
            val = bm.zeros(p.shape[0], dtype=bm.float64)
            return val
        
        def phi_source(phi_n, dt, t):
            
            def fphi(phi):
                tag0 = phi[:] > 1
                tag1 = (phi[:] >= -1) & (phi[:] <= 1)
                tag2 = phi[:] < -1
                f_val = phi.space.function()
                f_val[tag0] = 2/epsilon**2  * (phi[tag0] - 1)
                f_val[tag1] = (phi[tag1]**3 - phi[tag1]) / (epsilon**2)
                f_val[tag2] = 2/epsilon**2 * (phi[tag2] + 1)
                return f_val
            
            @barycentric
            def source(bcs, index):
                # phase_force = lambda p: phase_force(p, t)
                phi_val = phi_n(bcs, index)  
                result = (1+ dt*gamma/epsilon**2) * phi_val
                result -= gamma * dt * fphi(phi_n)(bcs, index)
                ps = mesh.bc_to_point(bcs, index)
                result += dt * phase_force(ps, t)
                return result
            
            return source
        
        bar_mu = min(mu0, mu1)

        def us_source(phi_n, phi, u_n, s_n, dt, t, mv):
            rho_val = rho(phi)
            rho_n = rho(phi_n)
            # velocity_force = lambda p: velocity_force(p, t)
            @barycentric
            def mon_source(bcs, index):
                gphi_n = phi_n.grad_value(bcs, index)
                result0 = bm.sqrt(rho_n(bcs, index)[..., None]) * bm.sqrt(rho_val(bcs, index)[..., None]) * u_n(bcs, index)
                result1 = dt * bar_mu * s_n.grad_value(bcs, index)
                result2 = lam/gamma * (phi(bcs, index) - phi_n(bcs, index))[..., None] * gphi_n
                mv_gardphi = bm.einsum('cqi,cqi->cq', gphi_n, mv(bcs, index))
                result3 = dt*lam/gamma * mv_gardphi[..., None] * gphi_n 
                result = result0 - result1 - result2 + result3
                return result
            
            @barycentric
            def force_source(bcs, index):
                ps = mesh.bc_to_point(bcs, index)
                result = dt * rho_val(bcs,index)[...,None] * velocity_force(ps, t)
                return result
            
            @barycentric
            def source(bcs, index):
                result0 = mon_source(bcs, index)
                result1 = force_source(bcs, index)
                result = result0 + result1
                return result
            return source


        equation = [{
            "ac_cm": gamma/epsilon**2,
            "ac_cd": gamma,
            "ac_cc": 1.0,
            "ac_source": phi_source,
            "ns_cusm": rho,
            "ns_cusphi": lam/gamma,
            "ns_cusv": mu,
            "ns_cusc": rho,
            "ns_cussource": us_source,
            "ns_cpsd": rho,
            "ns_cu1l": rho,
            "ns_cp1l": bar_mu
        }]

        x0 = [{
            "init_phase": init_phase,
            "init_velocity": init_velocity,
            "init_pressure": init_pressure
        }]

        return (equation, boundary, is_boundary, x0)
        

