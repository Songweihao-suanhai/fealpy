from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["GearBox"]

class GearBox(CNodeType):
    
    TITLE: str = "变速箱.数值离散"
    PATH: str = "simulation.discretization"
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title ="网格"),
    ]
    OUTPUT_SLOTS = [
        PortConf("stiffness", DataType.TENSOR, title="刚度矩阵S"),
        PortConf("mass", DataType.TENSOR, title="质量矩阵M"),
        PortConf("mesh", DataType.MESH, title = "网格"),
    ]


    
    @staticmethod
    def run(mesh):

        from ...fem import BilinearForm
        from ...fem import LinearElasticityIntegrator
        from ...fem import ScalarMassIntegrator as MassIntegrator
        from ...backend import backend_manager as bm
        from ...material import LinearElasticMaterial
        from ...functionspace import functionspace
        GD = mesh.geo_dimension()
        space = functionspace(mesh, ('Lagrange', 1), shape=(-1, GD))
        """
        Construct the linear system for the gearbox shell model.
        """
        for name, data in mesh.data['materials'].items():
            material = LinearElasticMaterial(name, **data) 

        bform = BilinearForm(space)
        
        integrator = LinearElasticityIntegrator(material)
        integrator.assembly.set('fast')
        bform.add_integrator(integrator)
        S = bform.assembly()

        bform = BilinearForm(space)
        integrator = MassIntegrator(material.density)
        bform.add_integrator(integrator)
        M = bform.assembly()

        S = S.to_scipy()
        M = M.to_scipy()

        S = (S + S.T)/2.0
        M = (M + M.T)/2.0

        NN = mesh.number_of_nodes()
        name = mesh.data['boundary_conditions'][0][0]
        nset = mesh.data.get_node_set(name)

        isFNode = bm.zeros(NN, dtype=bm.bool)
        isFNode[nset] = True
        redges, rnodes = mesh.data.get_rbe2_edge()
        isRNode = bm.zeros(NN, dtype=bm.bool)
        isRNode = bm.set_at(isRNode, rnodes, True)

        mesh.data.add_node_data('isFNode', isFNode)
        isFreeNode = ~(isFNode | isRNode)
        isFreeDof = bm.repeat(isFreeNode, 3)
        mesh.data.add_node_data('isFreeNode', isFreeNode)
        S0 = S[isFreeDof,:][:,isFreeDof]
        M0 = M[isFreeDof,:][:,isFreeDof]

        return S0,M0,mesh
