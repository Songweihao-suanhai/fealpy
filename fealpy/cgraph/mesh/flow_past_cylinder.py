from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["FlowPastCylinder2d"]

class FlowPastCylinder2d(CNodeType):
    TITLE: str = "二维圆柱绕流网格建模"
    PATH: str = "preprocess.mesher"
    INPUT_SLOTS= [
        PortConf("material", DataType.LIST, 0, default=None, title="材料属性"),
        PortConf("box", DataType.TEXT, 0, default=(0.0, 2.2, 0.0, 0.41), title="求解域"),
        PortConf("center", DataType.TEXT, 0, default=(0.2, 0.2), title="圆心坐标"),
        PortConf("radius", DataType.FLOAT, 0, default=0.05, title="圆柱半径"),
        PortConf("n_circle", DataType.INT, 0, default=100, title="圆柱周围点数"),
        PortConf("h", DataType.FLOAT, 0, default=0.01, title="全局网格尺寸")
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格")
    ]
    @staticmethod
    def run(material, box, center, radius, n_circle, h):
        import gmsh 
        import math
        from fealpy.backend import backend_manager as bm
        from fealpy.mesh import TriangleMesh 
        box = bm.tensor(eval(box, None, vars(math)), dtype=bm.float64)
        center = bm.tensor(eval(center, None, vars(math)), dtype=bm.float64)
        cx = center[0]
        cy = center[1] 
        gmsh.initialize() 
        gmsh.model.add("rectangle_with_polygon_hole") 
        xmin, xmax, ymin, ymax = box 
        p1 = gmsh.model.geo.addPoint(xmin, ymin, 0) 
        p2 = gmsh.model.geo.addPoint(xmax, ymin, 0) 
        p3 = gmsh.model.geo.addPoint(xmax, ymax, 0) 
        p4 = gmsh.model.geo.addPoint(xmin, ymax, 0) 
        l1 = gmsh.model.geo.addLine(p1, p2) 
        l2 = gmsh.model.geo.addLine(p2, p3) 
        l3 = gmsh.model.geo.addLine(p3, p4) 
        l4 = gmsh.model.geo.addLine(p4, p1) 
        outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4]) 
        theta = bm.linspace(0, 2*bm.pi, n_circle, endpoint=False) 
        circle_pts = [] 
        for t in theta:
            x = cx + radius * bm.cos(t) 
            y = cy + radius * bm.sin(t) 
            pid = gmsh.model.geo.addPoint(x, y, 0) 
            circle_pts.append(pid) 
        circle_lines = [] 
        for i in range(n_circle): 
            l = gmsh.model.geo.addLine(circle_pts[i], circle_pts[(i + 1) % n_circle]) 
            circle_lines.append(l) 
        circle_loop = gmsh.model.geo.addCurveLoop(circle_lines) 
        surf = gmsh.model.geo.addPlaneSurface([outer_loop, circle_loop]) 
        gmsh.model.geo.synchronize() 
        inlet = gmsh.model.addPhysicalGroup(1, [l4], tag = 1) 
        gmsh.model.setPhysicalName(1, 1, "inlet") 
        outlet = gmsh.model.addPhysicalGroup(1, [l2], tag = 2) 
        gmsh.model.setPhysicalName(1, 2, "outlet") 
        wall = gmsh.model.addPhysicalGroup(1, [l1, l3], tag = 3) 
        gmsh.model.setPhysicalName(1, 3, "walls") 
        cyl = gmsh.model.addPhysicalGroup(1, circle_lines, tag = 4) 
        gmsh.model.setPhysicalName(1, 4, "cylinder") 
        domain = gmsh.model.addPhysicalGroup(2, [surf], tag = 5) 
        gmsh.model.setPhysicalName(2, 5, "fluid") 
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h) 
        gmsh.model.mesh.generate(2) 
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes() 
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2) 
        tri_nodes = elem_node_tags[0].reshape(-1, 3) - 1 # 转为从0开始索引 
        node_coords = bm.array(node_coords).reshape(-1, 3)[:, :2] 
        tri_nodes = bm.array(tri_nodes, dtype=bm.int32) 
        boundary = [] 
        boundary_tags = [1, 2, 3, 4] 
        for tag in boundary_tags: 
            node_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, tag) # 转换为从 0 开始的索引 
            boundary.append(bm.array(node_tags - 1, dtype=bm.int32)) 
        boundary = boundary 
        gmsh.finalize() 
        mesh = TriangleMesh(node_coords, tri_nodes)
        mesh.box = box
        mesh.center = center

        eps = 1e-10
        def is_inlet_boundary(p):
            x = p[..., 0]
            return bm.abs(x - box[0]) < eps
        def is_outlet_boundary(p):
            x = p[...,0]
            y = p[...,1]
            cond1 = bm.abs(x - box[1]) < eps
            cond2 = bm.abs(y-box[2])>eps
            cond3 = bm.abs(y-box[3])>eps
            return (cond1) & (cond2 & cond3) 
        def is_wall_boundary(p):
            y = p[..., 1]
            return (bm.abs(y - box[2]) < eps) | (bm.abs(y - box[3]) < eps)
        
        def is_velocity_boundary(p):
            return ~is_outlet_boundary(p)
        
        def is_pressure_boundary(p = None):
            if p is None:
                return 1
            return is_outlet_boundary(p)
        
        mesh.is_inlet_boundary = is_inlet_boundary
        mesh.is_outlet_boundary = is_outlet_boundary
        mesh.is_wall_boundary = is_wall_boundary
        mesh.is_velocity_boundary = is_velocity_boundary
        mesh.is_pressure_boundary = is_pressure_boundary
        mesh.material = material

        if material is not None:

            material = material[0]
            NN = mesh.number_of_nodes()
            for k, value in material.items():
                setattr(mesh, k, value)
                if value is float:
                    mesh.nodedata[k] = material[k] * bm.ones((NN, ), dtype=bm.float64)

        return mesh

