from ..nodetype import CNodeType, PortConf, DataType

__all__ = ['MMGUACNSFEMModel']

class MMGUACNSFEMModel(CNodeType):
    
    TITLE: str = "ACNS 方程有限元计算模型(移动网格)"
    PATH: str = "simulation.solvers"
    DESC: str = """该节点实现了两相不可压流体的 Allen-Cahn-Navier-Stokes (ACNS) 方程组有限元求解
                器。ACNS 模型结合了相场法与流体力学方程，用以描述两种不可混溶流体的界面演化与流动耦合过程。
                通过时间步推进(dt, nt)，程序在每个时间步内依次执行：
                1. Allen-Cahn 方程：更新相场函数 φ 与化学势 μ；
                2. Navier-Stokes 方程：根据当前相场计算密度场 ρ(φ)，更新速度场 u 与压力场 p。

                输入参数：
                - dt (float)：时间步长；
                - nt (int)：总时间步数；
                - uspace (SpaceType)：速度场有限元空间；
                - pspace (SpaceType)：压力场有限元空间；
                - phispace (SpaceType)：相场 φ 的有限元空间；
                - update_ac (function)：组装并更新 Allen–Cahn 子问题；
                - update_us (function)：组装并更新辅助速度系统；
                - update_ps (function)：压力更新流程；
                - update_velocity (function)：速度校正/投影更新；
                - update_gauge (function)：规范变量（gauge）更新；
                - update_pressure (function)：压力更新函数；
                - q (int)：数值积分的求积阶数。

                输出结果：
                - u ：最终时刻的速度场；
                - ux、uy ：速度在 x、y 方向的分量；
                - p ：压力场；
                - phi ：相场函数。

                使用示例：
                可将“相场更新模块 (AC)”与“流体更新模块 (gu-NS)”分别连接到 `update_ac` 与 
                (`update_us`, `update_ps`, `update_velocity`, `update_gauge`, `update_p`)，
                设置好初始界面与网格信息后，即可在多时间步迭代中自动完成流体界面的演化与速度场的时序求解。
                """
    INPUT_SLOTS = [
        PortConf("i", DataType.INT, title="迭代步"),
        PortConf("dt", DataType.FLOAT, 0, title="时间步长", default=0.001),
        PortConf("phi", DataType.FUNCTION, title="相场函数"),
        PortConf("u", DataType.FUNCTION, title="速度场"),
        PortConf("p", DataType.FUNCTION, title="压力场"),
        PortConf("equation", DataType.LIST, title="方程"),
        PortConf("boundary", DataType.FUNCTION, title="边界条件"),
        PortConf("x0", DataType.LIST, title="初始值"),
        PortConf("xn", DataType.DICT, title="前置值"),
        
    ]
    OUTPUT_SLOTS = [
        PortConf("u", DataType.FUNCTION, title="速度场"),
        PortConf("p", DataType.FUNCTION, title="压力场"),
        PortConf("phi", DataType.FUNCTION, title="相场函数"),
        PortConf("xn", DataType.DICT, title="前置值"),
    ]
    
    @staticmethod
    def run(i, dt,
            phi, u, p,
            equation, boundary, 
            x0, xn):
        from fealpy.backend import bm
        from fealpy.solver import spsolve
        from fealpy.functionspace import TensorFunctionSpace

        uspace = u.space
        pspace = p.space
        phispace = phi.space
        velocity_force = equation.get('velocity_force')
        phase_force = equation.get('phase_force')
        init_phase = x0["init_phase"]
        init_velocity = x0["init_velocity"]
        init_pressure = x0["init_pressure"]
        velocity_dirichlet_bc = boundary['velocity']
        mesh = uspace.mesh
        # export_dir = Path(output_dir).expanduser().resolve()
        # export_dir.mkdir(parents=True, exist_ok=True)
        domain = mesh.box

        def set_move_mesher(mesh, phi_n , phispace,
                            mmesher:str = 'GFMMPDE',
                            beta :float = 1,
                            tau :float = 0.1,
                            tmax :float = 0.5,
                            alpha :float = 0.75,
                            moltimes :int = 4,
                            monitor: str = 'arc_length',
                            mol_meth :str = 'projector',
                            config : dict = None):
            
            from fealpy.mmesh.mmesher import MMesher
            mesh.meshdata['vertices'] =bm.array([[domain[0], domain[2]],
                             [domain[1], domain[2]],
                             [domain[1], domain[3]],
                             [domain[0], domain[3]]], dtype=bm.float64)
            mm = MMesher(mesh, 
                        uh = phi_n ,
                        space= phispace,
                        beta=beta,
                        ) 
            mm.config.active_method = mmesher
            mm.config.tau = tau
            mm.config.t_max = tmax
            mm.config.alpha = alpha
            mm.config.mol_times = moltimes
            mm.config.monitor = monitor
            mm.config.mol_meth = mol_meth
            mm.config.is_pre = False
            if config is not None:
                for key, value in config.items():
                    # check if the key exists in the config
                    getattr(mm.config, key)
                    # if it exists, set the value
                    setattr(mm.config, key, value)

            mm.initialize()
            mm.set_interpolation_method('linear')
            node_n = mesh.node.copy()
            smspace =  mm.instance.mspace
            mspace = TensorFunctionSpace(smspace, (mesh.GD,-1))
            mesh_velocity = mspace.function()
            return mm, mesh_velocity, node_n
        
        def save_vtu(step: int ,export_dir):
            mesh.nodedata['interface'] = phi
            mesh.nodedata['velocity'] = u.reshape(mesh.GD,-1).T
            mesh.nodedata['pressure'] = p
            fname = export_dir / f"two_phase_flow_{str(step).zfill(10)}.vtu"
            mesh.to_vtk(fname=fname)
                
        def compute_bubble_centroid():
            # 获取网格节点和相场值
            nodes = mesh.node  # 网格节点坐标
            cell = mesh.cell
            cell_to_dof = phispace.cell_to_dof()
            phi_cell_values = bm.mean(phi[cell_to_dof],axis=-1)  # 相场值

            # 筛选气泡区域（phi > 0）
            bubble_mask = phi_cell_values > 0
            bubble_bc_nodes = bm.mean(nodes[cell[bubble_mask]],axis=1)
            bubble_phi = phi_cell_values[bubble_mask]

            # 计算质心位置
            centroid = bm.sum(bubble_bc_nodes.T * bubble_phi, axis=1) / bm.sum(bubble_phi)
            return centroid
        
        # Initialize functions
        u = uspace.function()  # Current velocity
        u_n = uspace.function()  # Previous time velocity
        p = pspace.function()  # Current pressure
        phi = phispace.function()  # Current phase-field
        phi_n = phispace.function()  # Previous time phase-field
        
        # Intermediate variables
        us = uspace.function()  # Intermediate velocity
        s = pspace.function()  # guage variable
        s_n = pspace.function()  # guage variable
        ps = pspace.function()  # Intermediate pressure
        
        ugdof = uspace.number_of_global_dofs()
         
        # t = 0.0
        t  = dt * (i + 1)
        # Move mesh according to phase field
        if i == 0:
            # Set initial velocity
            u_n[:] = uspace.interpolate(lambda p: init_velocity(p))
            u[:] = u_n[:]
            # Set initial pressure
            p[:] = pspace.interpolate(lambda p: init_pressure(p))
            # Set initial phase-field
            phi_n[:] = phispace.interpolate(lambda p: init_phase(p))
            phi[:] = phi_n[:]

            mm, mesh_velocity, node_n = set_move_mesher(mesh, phi_n, phispace)
            # First time step, need to initialize the mesh movement

            mm.run()
            node_n = mesh.node.copy()
            phi_n = phispace.interpolate(lambda p: init_phase(p))
            phi[:] = phi_n[:]
            # save_vtu(step = 0,export_dir=export_dir)
        else:
            phi_n = xn["phi_n"]
            u_n = xn["u_n"]
            s_n = xn["s_n"]
            node_n = xn["node_n"]
            mesh_velocity = xn["mesh_velocity"]
            mm = xn["mm"]
            mm.run()
            
        mesh_velocity[:] = ((mesh.node - node_n)/dt).T.flatten()
        # Define time-dependent forces and boundary conditions
        current_v_force = lambda p: velocity_force(p, t)
        current_phi_force = lambda p: phase_force(p, t)
        current_v_dirichlet_bc = lambda p: velocity_dirichlet_bc(p, t)
        # Update phase-field using Allen-Cahn equation
        from fealpy.cgraph.cfd.fem import AllenCahn
        update_ac = AllenCahn.method(phispace, init_phase, q=4)
        from fealpy.cgraph.cfd.fem import GaugeUzawaNS
        update_us, update_ps, update_velocity, update_gauge, update_pressure = \
            GaugeUzawaNS.method(mesh, uspace, pspace, phispace, q=3)
        ac_A , ac_b = update_ac(u_n, phi_n, dt,current_phi_force, mesh_velocity)   # 此处差一个网格速度
        phi_val = spsolve(ac_A, ac_b,solver= 'scipy')
        phi[:] = phi_val[:-1]
        print(phi)

        print("theta:",phi_val[-1])
        # Update auxiliary velocity field
        us_A , us_b = update_us(phi_n , phi , u_n ,s_n ,dt,
                                current_v_force,current_v_dirichlet_bc,mesh_velocity)
        us[:] = spsolve(us_A, us_b,solver= 'scipy')
        # Update intermediate pressure
        ps_A , ps_b = update_ps(phi, us)
        ps[:] = spsolve(ps_A, ps_b,solver= 'scipy')
        # Update velocity field           
        u_A , u_b = update_velocity(phi, us, ps)
        u[:] = spsolve(u_A, u_b,solver= 'scipy')
        # Update gauge variable
        s_A , s_b = update_gauge(s_n, us)
        s[:] = spsolve(s_A, s_b,solver= 'scipy')
        # Update pressure field
        p_A , p_b = update_pressure(s, ps, dt)
        p[:] = spsolve(p_A, p_b,solver= 'scipy')
                    
        # Prepare for next time step
        phi_n[:] = phi[:]
        u_n[:] = u[:]
        s_n[:] = s[:]
        node_n = mesh.node.copy()
        mm.instance.uh = phi
        xn = {
            "phi_n": phi_n,
            "u_n": u_n,
            "s_n": s_n,
            "node_n": node_n,
            "mesh_velocity": mesh_velocity,
            "mm": mm
        }
        
        # Save results
        # save_vtu(i+1,export_dir)
        # Compute and print bubble centroid
        centroid = compute_bubble_centroid()
        print(f"Time step {i+1}, Time {t:.4f}, Bubble Centroid: {centroid}")
            
        return u, p, phi, xn