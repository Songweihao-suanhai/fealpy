
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["StokesMathematics"]

class StokesMathematics(CNodeType):
    TITLE: str = "Stokes数学模型"
    PATH: str = "preprocess.modeling"
    INPUT_SLOTS = [
        PortConf("u", DataType.TENSOR, title="速度"),
        PortConf("p", DataType.TENSOR, title="压力")
    ]
    OUTPUT_SLOTS = [
        PortConf("equation", DataType.LIST, title="方程"),
        PortConf("boundary", DataType.LIST, title="边界条件"),
        PortConf("is_boundary", DataType.LIST, title="边界标记")
    ]

    @staticmethod
    def run(u, p):
        from fealpy.backend import backend_manager as bm
        from fealpy.backend import TensorLike
        from fealpy.decorator import cartesian

        class DLD3D():
            def __init__(self, mesh):
                self.eps = 1e-10
                mesher = mesh.mesher
                self.thickness = mesher.options.get("thickness")
                self.radius = mesher.radius
                self.centers = mesher.centers
                self.inlet_boundary = mesher.inlet_boundary
                self.outlet_boundary = mesher.outlet_boundary
                self.wall_boundary = mesher.wall_boundary

            def get_dimension(self) -> int: 
                """Return the geometric dimension of the domain."""
                return 3

            @cartesian
            def source(self, p: TensorLike) -> TensorLike:
                """Compute exact source """
                x = p[..., 0]
                y = p[..., 1]
                z = p[..., 2]
                result = bm.zeros(p.shape, dtype=bm.float64)
                return result

            @cartesian
            def inlet_velocity(self, p: TensorLike) -> TensorLike:
                """Compute exact solution of velocity."""
                x = p[..., 0]
                y = p[..., 1]
                z = p[..., 2]
                result = bm.zeros(p.shape, dtype=bm.float64)
                result[..., 0] = 16*0.45*y*z*(1-y)*(0.1-z)/(0.41**4)
                return result
            
            @cartesian
            def outlet_pressure(self, p: TensorLike) -> TensorLike:
                """Compute exact solution of velocity."""
                x = p[..., 0]
                y = p[..., 1]
                z = p[..., 2]
                result = bm.zeros(p.shape[0], dtype=bm.float64)
                return result
            
            @cartesian
            def is_inlet_boundary(self, p: TensorLike) -> TensorLike:
                """Check if point where velocity is defined is on boundary."""
                bd = self.inlet_boundary
                return self.is_lateral_boundary(p, bd)  

            @cartesian
            def is_outlet_boundary(self, p: TensorLike) -> TensorLike:
                """Check if point where pressure is defined is on boundary."""
                bd = self.outlet_boundary
                return self.is_lateral_boundary(p, bd)
            
            @cartesian
            def is_wall_boundary(self, p: TensorLike) -> TensorLike:
                """Check if point where velocity is defined is on boundary."""
                bd = self.wall_boundary
                return self.is_lateral_boundary(p, bd)
            
            @cartesian
            def is_top_or_bottom(self, p: TensorLike) -> TensorLike:
                """Check if point where velocity is defined is on top or bottom boundary."""
                atol = 1e-12
                thickness = self.thickness
                cond = (bm.abs(p[:, -1]) < atol) | (bm.abs(p[:, -1] - thickness) < atol)
                return cond
            
            @cartesian
            def is_obstacle_boundary(self, p: TensorLike) -> TensorLike:
                """Check if point where velocity is defined is on boundary."""
                x = p[..., 0]
                y = p[..., 1]
                radius = self.radius
                atol = 1e-12
                on_boundary = bm.zeros_like(x, dtype=bool)
                for center in self.centers:
                    cx, cy = center
                    on_boundary |= (x - cx)**2 + (y - cy)**2 < radius**2 + atol
                return on_boundary
            
            @cartesian
            def is_velocity_boundary(self, p: TensorLike) -> TensorLike:
                """Check if point where velocity is defined is on boundary."""
                inlet = self.is_inlet_boundary(p)
                wall = self.is_wall_boundary(p)
                top_or_bottom = self.is_top_or_bottom(p)
                obstacle = self.is_obstacle_boundary(p)
                return inlet | wall | top_or_bottom | obstacle

            @cartesian
            def is_pressure_boundary(self, p: TensorLike = None) -> TensorLike:
                """Check if point where pressure is defined is on boundary."""
                if p is None:
                    return 1
                return self.is_outlet_boundary(p)
        
            @cartesian
            def velocity_dirichlet(self, p: TensorLike) -> TensorLike:
                """Optional: prescribed velocity on boundary, if needed explicitly."""
                inlet = self.inlet_velocity(p)
                is_inlet = self.is_inlet_boundary(p)
            
                result = bm.zeros_like(p, dtype=p.dtype)
                result[is_inlet] = inlet[is_inlet]
                
                return result
            
            @cartesian
            def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
                """Optional: prescribed pressure on boundary (usually for stability)."""
                return self.outlet_pressure(p)
            
            def is_lateral_boundary(self, p: TensorLike, bd: TensorLike) -> TensorLike:
                """Check if point is on boundary."""
                atol = 1e-12
                v0 = p[:, None, :-1] - bd[None, 0::2, :] # (NN, NI, 2)
                v1 = p[:, None, :-1] - bd[None, 1::2, :] # (NN, NI, 2)

                cross = v0[..., 0]*v1[..., 1] - v0[..., 1]*v1[..., 0] # (NN, NI)
                dot = bm.einsum('ijk,ijk->ij', v0, v1) # (NN, NI)
                cond = (bm.abs(cross) < atol) & (dot < atol)
                return bm.any(cond, axis=1)
        
        mesh = u.space.mesh
        model = DLD3D(mesh)

        equation = [{
            "diffusion" : mesh.mu,
            "pressure" : -1.0,
            "source" : model.source
        }]
        boundary = [{
            "velocity_boundary" : model.velocity_dirichlet,
            "pressure_boundary" : model.pressure_dirichlet
        }]
        is_boundary = [{
            "is_velocity_boundary" : model.is_velocity_boundary,
            "is_pressure_boundary" : model.is_pressure_boundary
        }]
    
        return (equation, boundary, is_boundary)

