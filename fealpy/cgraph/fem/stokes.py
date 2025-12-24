
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["StokesFEMModel"]


class StokesFEMModel(CNodeType):

    TITLE: str = "Stokes 方程离散"
    PATH: str = "simulation.discretization"
    INPUT_SLOTS = [
        PortConf("equation", DataType.LIST, title="方程"),
        PortConf("boundary", DataType.LIST, title="边界条件"),
        PortConf("is_boundary", DataType.LIST, title="边界"),
        PortConf("u", DataType.TENSOR, title="速度"),
        PortConf("p", DataType.TENSOR, title="压力")
    ]
    OUTPUT_SLOTS = [
        PortConf("bform", DataType.TENSOR, title="算子"),
        PortConf("lform", DataType.TENSOR, title="向量")
    ]

    @staticmethod
    def run(equation, boundary, is_boundary, u, p):
        from fealpy.fem import LinearForm, BilinearForm, BlockForm, LinearBlockForm
        from fealpy.fem import ScalarDiffusionIntegrator as DiffusionIntegrator
        from fealpy.fem import PressWorkIntegrator

        equation = equation[0]
        bd = boundary[0]
        is_boundary = is_boundary[0]
        uspace = u.space
        pspace = p.space
        velocity_dirichlet = bd["velocity_boundary"]
        pressure_dirichlet = bd["pressure_boundary"]
        is_velocity_boundary = is_boundary["is_velocity_boundary"]
        is_pressure_boundary = is_boundary["is_pressure_boundary"]

        A00 = BilinearForm(uspace)
        BD = DiffusionIntegrator()
        BD.coef = equation["diffusion"]
        A00.add_integrator(BD)
        A01 = BilinearForm((pspace, uspace))
        BP = PressWorkIntegrator()
        BP.coef = equation["pressure"]
        A01.add_integrator(BP)
        bform = BlockForm([[A00, A01], [A01.T, None]])

        L0 = LinearForm(uspace)
        L1 = LinearForm(pspace)
        lform = LinearBlockForm([L0, L1])

        from fealpy.fem import DirichletBC
        A = bform.assembly()
        F = lform.assembly()
        BC = DirichletBC(
            (uspace, pspace), 
            gd=(velocity_dirichlet, pressure_dirichlet), 
            threshold=(is_velocity_boundary, is_pressure_boundary),
            method='interp')
        A, F = BC.apply(A, F)

        return A, F
    