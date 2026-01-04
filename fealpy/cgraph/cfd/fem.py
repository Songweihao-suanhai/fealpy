from fealpy.backend import backend_manager as bm
from fealpy.decorator import barycentric, cartesian, variantmethod
from fealpy.fem import (BilinearForm, ScalarMassIntegrator, ScalarDiffusionIntegrator, BlockForm,
                        FluidBoundaryFrictionIntegrator,ScalarConvectionIntegrator, ViscousWorkIntegrator,
                        PressWorkIntegrator, LinearBlockForm, LinearForm, SourceIntegrator, DirichletBC)
from fealpy.sparse import COOTensor

class FEMBase:
    def lagrange_multiplier(pspace, A, b, c=0):
        """
        Constructs the augmented system matrix for Lagrange multipliers.
        c is the integral of pressure, default is 0.
        """
        LagLinearForm = LinearForm(pspace)
        LagLinearForm.add_integrator(SourceIntegrator(source=1))
        LagA = LagLinearForm.assembly()

        A1 = COOTensor(bm.array([bm.zeros(len(LagA), dtype=bm.int32),
                                bm.arange(len(LagA), dtype=bm.int32)]), LagA, spshape=(1, len(LagA)))

        A = BlockForm([[A, A1.T], [A1, None]])
        A = A.assembly_sparse_matrix(format='csr')
        b0 = bm.array([c])
        b  = bm.concatenate([b, b0], axis=0)
        return A, b
    
    def apply_bc(space, dirichlet, is_boundary , t):
        r"""Dirichlet boundary conditions for unsteady Navier-Stokes equations."""
        BC = DirichletBC(space=space,
                gd = cartesian(lambda p : dirichlet(p, t)),
                threshold = is_boundary,
                method = 'interp')
        return BC.apply


class IncompressibleNS(FEMBase):

    @variantmethod("IPCS")
    def method(self, uspace, pspace, 
               boundary: dict, 
               q: int , 
               apply_bc: callable = None):

        velocity_dirichlet = boundary["velocity"]
        pressure_dirichlet = boundary["pressure"]

        mesh = uspace.mesh
        is_velocity_boundary = mesh.geo.is_velocity_boundary
        is_pressure_boundary = mesh.geo.is_pressure_boundary

        if apply_bc is None:
            apply_bc = FEMBase.apply_bc
        
        #预测速度左端项
        predict_Bform = BilinearForm(uspace)
        predict_BM = ScalarMassIntegrator(q=q)  
        predict_Bform.add_integrator(predict_BM)
        
        if is_pressure_boundary() != 0:
            predict_BF = FluidBoundaryFrictionIntegrator(q=q, threshold=is_pressure_boundary)
            predict_Bform.add_integrator(predict_BF)
        
        predict_BVW = ScalarDiffusionIntegrator(q=q)
        predict_Bform.add_integrator(predict_BVW)

        #预测速度右端项
        from fealpy.fem import (LinearForm, SourceIntegrator, GradSourceIntegrator, 
                                BoundaryFaceSourceIntegrator)
        predict_Lform = LinearForm(uspace) 
        predict_LS = SourceIntegrator(q=q)
        predict_LS_f = SourceIntegrator(q=q)
        predict_LGS = GradSourceIntegrator(q=q)
        
        predict_Lform.add_integrator(predict_LS)
        predict_Lform.add_integrator(predict_LGS)
        predict_Lform.add_integrator(predict_LS_f)
        if is_pressure_boundary() != 0:
            predict_LBFS = BoundaryFaceSourceIntegrator(q=q, threshold=is_pressure_boundary)
            predict_Lform.add_integrator(predict_LBFS)

        #预测速度更新函数
        def predict_velocity_update(u0, p0, dt, ctd, cc, pc, cv, cbf): 
            mesh = uspace.mesh
            
            predict_BM.coef = ctd/dt
            predict_BVW.coef = cv
            
            @barycentric
            def LS_coef(bcs, index):
                masscoef = ctd(bcs, index)[..., bm.newaxis] if callable(ctd) else ctd
                result = 1/dt*masscoef*u0(bcs, index)
                ccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
                result -= ccoef * bm.einsum('...j, ...ij -> ...i', u0(bcs, index), u0.grad_value(bcs, index))
                return result

            @barycentric
            def LGS_coef(bcs, index):
                I = bm.eye(mesh.GD)
                result = bm.repeat(p0(bcs,index)[...,bm.newaxis], mesh.GD, axis=-1)
                result = bm.expand_dims(result, axis=-1) * I
                result *= pc(bcs, index) if callable(pc) else pc
                return result
            
            
            @barycentric
            def LBFS_coef(bcs, index):
                result = -bm.einsum('...i, ...j->...ij', p0(bcs, index), mesh.face_unit_normal(index=index))
                result *= pc(bcs, index) if callable(pc) else pc
                return result
            
            predict_LS_f.source = cbf
            predict_LS.source = LS_coef
            predict_LGS.source = LGS_coef
            if is_pressure_boundary() != 0:
                predict_BF.coef = -cv
                predict_LBFS.source = LBFS_coef

        #预测速度方程线性系统组装
        def predict_velocity(u0, p0, t, dt, ctd, cc, pc, cv, cbf): 
            Bform = predict_Bform
            Lform = predict_Lform
            predict_velocity_update(u0, p0, dt, ctd, cc, pc, cv, cbf)
            A = Bform.assembly()
            b = Lform.assembly()
            apply_bcu = apply_bc(uspace, velocity_dirichlet, is_velocity_boundary, t)
            A, b = apply_bcu(A, b)
            return A, b 
            
        #压力修正左端项
        pressure_Bform = BilinearForm(pspace)
        pressure_BD = ScalarDiffusionIntegrator(q=q)
        pressure_Bform.add_integrator(pressure_BD) 

        #压力修正右端项
        pressure_Lform = LinearForm(pspace)
        pressure_LS = SourceIntegrator(q=q)
        pressure_LGS = GradSourceIntegrator(q=q)
        
        pressure_Lform.add_integrator(pressure_LS)
        pressure_Lform.add_integrator(pressure_LGS)

        #压力修正更新函数
        def pressure_update(us, p0, dt, ctd, pc):
            
            pressure_BD.coef = pc

            @barycentric
            def LS_coef(bcs, index=None):
                result = -1/dt*bm.trace(us.grad_value(bcs, index), axis1=-2, axis2=-1)
                result *= ctd(bcs, index) if callable(ctd) else ctd
                return result
            pressure_LS.source = LS_coef
            
            @barycentric
            def LGS_coef(bcs, index=None):
                result = p0.grad_value(bcs, index)
                result *= pc(bcs, index) if callable(pc) else pc
                return result
            pressure_LGS.source = LGS_coef

        #压力修正方程线性系统组装
        def correct_pressure(us, p0, t, dt, ctd, pc):
            Bform = pressure_Bform
            Lform = pressure_Lform
            pressure_update(us, p0, dt, ctd, pc)
            A = Bform.assembly()
            b = Lform.assembly()
            if is_pressure_boundary() == 0:
                A, b = FEMBase.lagrange_multiplier(pspace, A, b, 0)
            else:
                apply_bcp = apply_bc(pspace, pressure_dirichlet, is_pressure_boundary, t)
                A, b = apply_bcp(A, b)
            return A, b
            
        #速度修正左端项
        correct_Bform = BilinearForm(uspace)
        correct_BM = ScalarMassIntegrator(q=q)
        correct_Bform.add_integrator(correct_BM)

        #速度修正右端项
        correct_Lform = LinearForm(uspace)
        correct_LS = SourceIntegrator(q=q)
        correct_Lform.add_integrator(correct_LS)

        #速度修正更新函数
        def correct_velocity_update(us, p0, p1, dt, ctd):

            correct_BM.coef = ctd
            @barycentric
            def BM_coef(bcs, index):
                masscoef = ctd(bcs, index)[..., bm.newaxis] if callable(ctd) else ctd
                result = masscoef * us(bcs, index)
                result -= dt*(p1.grad_value(bcs, index) - p0.grad_value(bcs, index))
                return result
            correct_LS.source = BM_coef

        #速度修正方程线性系统组装
        def correct_velocity(us, p0, p1, t, dt, ctd):
            """速度校正"""
            Bform = correct_Bform
            Lform = correct_Lform
            correct_velocity_update(us, p0, p1, dt, ctd)
            A = Bform.assembly()
            b = Lform.assembly()
            apply_bcu = apply_bc(uspace, velocity_dirichlet, is_velocity_boundary, t)
            A, b = apply_bcu(A, b)
            return A, b
            
        return predict_velocity, correct_pressure, correct_velocity
    
    @method.register("BDF2")
    def method(uspace, pspace, 
               boundary_condition = None, 
               is_boundary = None, 
               q = 3, 
               apply_bc = None):  

        def update(u_0, u_1, dt, t, ctd, cc, pc, cv, cbf, apply_bc = apply_bc):
            
            ## BilinearForm
            A00 = BilinearForm(uspace)
            BM = ScalarMassIntegrator(q=q)
            BM.coef = 3*ctd/(2*dt)
            BC = ScalarConvectionIntegrator(q=q)
            def BC_coef(bcs, index): 
                ccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
                result = 2* ccoef * u_1(bcs, index)
                return result
            BC.coef = BC_coef
            BD = ViscousWorkIntegrator(q=q)
            BD.coef = 2*cv 

            A00.add_integrator(BM)
            A00.add_integrator(BC)
            A00.add_integrator(BD)

            A01 = BilinearForm((pspace, uspace))
            BPW0 = PressWorkIntegrator(q=q)
            BPW0.coef = -pc
            A01.add_integrator(BPW0) 

            A10 = BilinearForm((pspace, uspace))
            BPW1 = PressWorkIntegrator(q=q)
            BPW1.coef = -1
            A10.add_integrator(BPW1)
            
            A = BlockForm([[A00, A01], [A10.T, None]])

            ## LinearForm
            L0 = LinearForm(uspace) 
            LSI_U = SourceIntegrator(q=q)
            @barycentric
            def LSI_U_coef(bcs, index):
                masscoef = ctd(bcs, index)[..., bm.newaxis] if callable(ctd) else ctd
                result0 =  masscoef * (4*u_1(bcs, index) - u_0(bcs, index)) / (2*dt)
                
                ccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
                result1 = ccoef*bm.einsum('cqij, cqj->cqi', u_1.grad_value(bcs, index), u_0(bcs, index))
                cbfcoef = cbf(bcs, index) if callable(cbf) else cbf
                
                result = result0 + result1 + cbfcoef
                return result
            LSI_U.source = LSI_U_coef
            L0.add_integrator(LSI_U)

            L1 = LinearForm(pspace)
            L = LinearBlockForm([L0, L1])

            A = A.assembly()
            L = L.assembly()
            if apply_bc is None:
                (velocity_dirichlet, pressure_dirichlet) = boundary_condition()
                (is_velocity_boundary, is_pressure_boundary) = is_boundary()
                gd_v = cartesian(lambda p: velocity_dirichlet(p, t))
                gd_p = cartesian(lambda p: pressure_dirichlet(p, t))
                gd = (gd_v, gd_p)
                is_bd = (is_velocity_boundary, is_pressure_boundary)
                apply = FEMBase.apply_bc((uspace, pspace), gd, is_bd, t)
                A, L = apply(A, L)
            else:
                A, L = apply_bc(A, L)
            return A, L
        
        return update


class CahnHilliard(FEMBase):
    def method(phispace, q, s):

        def update(u_0, u_1, phi_0, phi_1, dt, cm, ci, cf):

            A00 = BilinearForm(phispace)
            BM_phi = ScalarMassIntegrator(q=q)
            BM_phi.coef = 3/(2*dt)
            
            BC_phi = ScalarConvectionIntegrator(q=q) 
            BC_phi.coef = 2*u_1

            A00.add_integrator(BM_phi)
            A00.add_integrator(BC_phi)

            A01 = BilinearForm(phispace)
            BD_phi = ScalarDiffusionIntegrator(q=q)
            BD_phi.coef = cm
            
            A01.add_integrator(BD_phi)
            
            A10 = BilinearForm(phispace)
            BD_mu = ScalarDiffusionIntegrator(q=q)
            BD_mu.coef = -ci

            BM_mu0 = ScalarMassIntegrator(q=q)
            BM_mu0.coef = -s * cf

            A10.add_integrator(BD_mu)
            A10.add_integrator(BM_mu0)

            A11 = BilinearForm(phispace)
            BM_mu1 = ScalarMassIntegrator(q=q)
            BM_mu1.coef = 1

            A11.add_integrator(BM_mu1)  

            A = BlockForm([[A00, A01], [A10, A11]]) 


            L0 = LinearForm(phispace)
            LS_phi = SourceIntegrator(q=q)
            @barycentric
            def LS_phi_coef(bcs, index):
                result = (4*phi_1(bcs, index) - phi_0(bcs, index))/(2*dt)
                result += bm.einsum('jid, jid->ji', u_0(bcs, index), phi_1.grad_value(bcs, index))
                return result

            LS_phi.source = LS_phi_coef
            L0.add_integrator(LS_phi)

            L1 = LinearForm(phispace)
            LS_mu = SourceIntegrator(q=q)
            @barycentric
            def LS_mu_coef(bcs, index): 
                result = -2*(1+s)*phi_1(bcs, index) + (1+s)*phi_0(bcs, index)
                result += 2*phi_1(bcs, index)**3 - phi_0(bcs, index)**3
                result *= cf
                return result
            LS_mu.source = LS_mu_coef

            L1.add_integrator(LS_mu)

            L = LinearBlockForm([L0, L1])
            A = A.assembly()
            L = L.assembly()
            return A, L
        
        return update
    

class AllenCahn(FEMBase):
    def method(phispace, init_phase, q):
        mesh = phispace.mesh
        init_mass = mesh.integral(init_phase)
        gamma = mesh.gamma
        epsilon = mesh.epsilon
        d = 0.005
        epsilon /= d


        def update_ac(u_n, phi_n, dt,phase_force, mv: None):
            # 移动网格行为产生的速度场 mv 可选
            if mv is None:
                def mv(bcs):
                    NC = mesh.number_of_cells()
                    GD = mesh.geo_dimension()
                    shape = (NC, bcs.shape[0], GD)
                    return bm.zeros(shape, dtype=bm.float64)
                
            bform = BilinearForm(phispace)
            lform = LinearForm(phispace)
            
            SMI = ScalarMassIntegrator(coef=1.0+dt*(gamma/epsilon**2), q=q)
            SDI = ScalarDiffusionIntegrator(coef=dt*gamma, q=q)
            SCI = ScalarConvectionIntegrator(q=q)
            SSI = SourceIntegrator(q=q)
            
            bform.add_integrator(SMI, SDI, SCI)
            lform.add_integrator(SSI)
            
            @barycentric
            def convection_coef(bcs, index):
                result = dt * (u_n(bcs, index) - mv(bcs, index))
                return result
            
            def fphi(phi):
                tag0 = phi[:] > 1
                tag1 = (phi[:] >= -1) & (phi[:] <= 1)
                tag2 = phi[:] < -1
                f_val = phispace.function()
                f_val[tag0] = 2/epsilon**2  * (phi[tag0] - 1)
                f_val[tag1] = (phi[tag1]**3 - phi[tag1]) / (epsilon**2)
                f_val[tag2] = 2/epsilon**2 * (phi[tag2] + 1)
                return f_val
            
            @barycentric
            def source(bcs , index):
                phi_val = phi_n(bcs, index)  
                result = (1+ dt*gamma/epsilon**2) * phi_val
                result -= gamma * dt * fphi(phi_n)(bcs, index)
                ps = mesh.bc_to_point(bcs, index)
                result += dt * phase_force(ps)
                return result
            
            SCI.coef = convection_coef
            SSI.source = source
            A = bform.assembly()
            b = lform.assembly()

            def lagrange_multiplier(A, b):
                LagLinearForm = LinearForm(phispace)
                Lag_SSI = SourceIntegrator(source=1, q=q)
                LagLinearForm.add_integrator(Lag_SSI)
                LagA = LagLinearForm.assembly()
                A0 = -dt * gamma * bm.ones(phispace.number_of_global_dofs())
                A0 = COOTensor(bm.array([bm.arange(len(A0), dtype=bm.int32), 
                                        bm.zeros(len(A0), dtype=bm.int32)]), A0, spshape=(len(A0), 1))
                A1 = COOTensor(bm.array([bm.zeros(len(LagA), dtype=bm.int32),
                                        bm.arange(len(LagA), dtype=bm.int32)]), LagA,
                                        spshape=(1, len(LagA)))
                b0 = bm.array([init_mass], dtype=bm.float64)
                A_block = BlockForm([[A, A0], [A1, None]])
                A_block = A_block.assembly_sparse_matrix(format='csr')
                b_block = bm.concat([b, b0], axis=0)
                return A_block, b_block
            
            A_block, b_block = lagrange_multiplier(A, b)
            return A_block, b_block
        return update_ac


class GaugeUzawaNS(FEMBase):
    def method(mesh, uspace, pspace, phispace, q):
        mesh = uspace.mesh
        mu0 = mesh.mu0
        mu1 = mesh.mu1
        rho0 = mesh.rho0
        rho1 = mesh.rho1
        lam = mesh.lam
        gamma = mesh.gamma
        bar_mu = min(mu0, mu1)

        d = 0.005
        g = 9.8
        ref_length = d
        ref_velocity = (g*d)**0.5
        ref_rho = min(rho0,rho1)
        ref_mu = ref_rho*ref_length*ref_velocity
        
        box = mesh.box
        # epsilon /= ref_length
        
        rho0 /= ref_rho
        rho1 /= ref_rho
        mu0 /= ref_mu
        mu1 /= ref_mu
        
        d /= ref_length
        g = 1.0

        us_bform = BilinearForm(uspace)
        ps_bform = BilinearForm(pspace)
        u_bform = BilinearForm(uspace)
        s_bform = BilinearForm(pspace)
        p_bform = BilinearForm(pspace)
        
        us_lform = LinearForm(uspace)
        ps_lform = LinearForm(pspace)
        u_lform = LinearForm(uspace)
        s_lform = LinearForm(pspace)
        p_lform = LinearForm(pspace)
        
        us_SMI = ScalarMassIntegrator(q=q)
        us_SMI_phiphi = ScalarMassIntegrator(q=q)
        us_VWI = ViscousWorkIntegrator(q=q)
        us_SCI = ScalarConvectionIntegrator(q=q)
        us_SSI = SourceIntegrator(q=q)
        
        ps_SDI = ScalarDiffusionIntegrator(q=q)
        ps_SSI = SourceIntegrator(q=q)
        
        u_SMI = ScalarMassIntegrator(coef=1.0 , q=q)
        u_SSI = SourceIntegrator(q=q)
        
        s_SMI = ScalarMassIntegrator(coef=1.0 , q=q)
        s_SSI = SourceIntegrator(q=q)
        
        p_SMI = ScalarMassIntegrator(coef=1.0+1e-8 , q=q)
        p_SSI = SourceIntegrator(q=q)
        
        us_bform.add_integrator(us_SMI, us_VWI, us_SCI , us_SMI_phiphi)
        us_lform.add_integrator(us_SSI)
        ps_bform.add_integrator(ps_SDI)
        ps_lform.add_integrator(ps_SSI)
        u_bform.add_integrator(u_SMI)
        u_lform.add_integrator(u_SSI)
        s_bform.add_integrator(s_SMI)
        s_lform.add_integrator(s_SSI)
        p_bform.add_integrator(p_SMI)
        p_lform.add_integrator(p_SSI)
        
        bc = DirichletBC(uspace)
        
        def density(phi):
            tag0 = phi[:] >1
            tag1 = phi[:] < -1
            phi[tag0] = 1
            phi[tag1] = -1
            rho = phispace.function()
            rho[:] = 0.5 * (rho0 + rho1) + 0.5 * (rho0 - rho1) * phi
            return rho
        
        def viscosity(phi):
            tag0 = phi[:] >1
            tag1 = phi[:] < -1
            phi[tag0] = 1
            phi[tag1] = -1
            mu = phispace.function()
            mu[:] = 0.5 * (mu0 + mu1) + 0.5 * (mu0 - mu1) * phi
            return mu
        
        def update_us(phi_n , phi , u_n ,s_n ,dt,
                      velocity_force,velocity_dirichlet_bc, mv = None):
            # 移动网格行为产生的速度场 mv 可选
            if mv is None:
                def mv(bcs):
                    NC = mesh.number_of_cells()
                    GD = mesh.geo_dimension()
                    shape = (NC, bcs.shape[0], GD)
                    return bm.zeros(shape, dtype=bm.float64)

            mu = viscosity(phi)
            rho = density(phi)
            rho_n = density(phi_n)

            @barycentric
            def mass_coef(bcs,index):
                uh0_val = u_n(bcs, index)
                guh0_val = u_n.grad_value(bcs, index)
                rho1_val = rho(bcs, index)
                grho1_val = rho.grad_value(bcs, index)
                div_u_rho = bm.einsum('cqii,cq->cq', guh0_val, rho1_val) 
                div_u_rho += bm.einsum('cqd,cqd->cq', grho1_val, uh0_val) 
                result0 = rho1_val + 0.5 * dt * div_u_rho
                
                # 添加移动网格项
                result1 = 0.5*dt*bm.einsum('cqi,cqi->cq', grho1_val, mv(bcs, index))
                result = result0 - result1
                return result
            
            @barycentric
            def phiphi_mass_coef(bcs, index):
                gphi_n = phi_n.grad_value(bcs, index)
                gphi_gphi = bm.einsum('cqi,cqj->cqij', gphi_n, gphi_n)
                result = (lam * dt/gamma) * gphi_gphi
                return result
            
            @barycentric
            def convection_coef(bcs, index):
                uh_n_val = u_n(bcs, index)
                rho1_val = rho(bcs, index)
                result = rho1_val[...,None] * (uh_n_val - mv(bcs, index))
                return dt * result
            
            @barycentric
            def mon_source(bcs, index):
                gphi_n = phi_n.grad_value(bcs, index)
                result0 = bm.sqrt(rho_n(bcs, index)[..., None]) * bm.sqrt(rho(bcs, index)[..., None]) * u_n(bcs, index)
                result1 = dt * bar_mu * s_n.grad_value(bcs, index)
                result2 = lam/gamma * (phi(bcs, index) - phi_n(bcs, index))[..., None] * gphi_n
                mv_gardphi = bm.einsum('cqi,cqi->cq', gphi_n, mv(bcs, index))
                result3 = dt*lam/gamma * mv_gardphi[..., None] * gphi_n 
                result = result0 - result1 - result2 + result3
                return result
            
            @barycentric
            def force_source(bcs, index):
                ps = mesh.bc_to_point(bcs, index)
                result = dt * rho(bcs,index)[...,None] * velocity_force(ps)
                return result
            
            @barycentric
            def source(bcs, index):
                result0 = mon_source(bcs, index)
                result1 = force_source(bcs, index)
                result = result0 + result1
                return result
            
            us_SMI.coef = mass_coef
            us_SMI_phiphi.coef = phiphi_mass_coef
            us_VWI.coef = dt * mu
            us_SCI.coef = convection_coef
            us_SSI.source = source
                        
            A = us_bform.assembly()
            b = us_lform.assembly()
            
            bc.gd = velocity_dirichlet_bc
            A , b = bc.apply(A, b)
            return A , b
        
        def update_ps(phi, us):
            rho = density(phi)
            
            @barycentric
            def diffusion_coef(bcs, index):
                result = 1 / rho(bcs, index)
                return result
            
            @barycentric
            def source_coef(bcs, index):
                uh_grad_val = us.grad_value(bcs, index)
                div_u = bm.einsum('cqii->cq', uh_grad_val)
                return div_u
            
            ps_SDI.coef = diffusion_coef
            ps_SSI.source = source_coef
            
            A = ps_bform.assembly()
            b = ps_lform.assembly()
            return A , b
        
        def update_velocity(phi, us, ps):
            rho = density(phi)
            
            @barycentric
            def source_coef(bcs, index):
                result = us(bcs, index)
                result += (1/rho(bcs, index)[..., None] )* ps.grad_value(bcs, index)
                return result
            
            u_SSI.source = source_coef
            
            A = u_bform.assembly()
            b = u_lform.assembly()
            return A , b
        
        def update_gauge(s_n, us):
            @barycentric
            def source_coef(bcs,index):
                result = s_n(bcs,index) - bm.einsum('cqii->cq', us.grad_value(bcs,index))
                return result
            s_SSI.source = source_coef

            A = s_bform.assembly()
            b = s_lform.assembly()
            return A , b
        
        def update_pressure(s, ps, dt):
            @barycentric
            def source_coef(bcs, index):
                result = -1/dt * ps(bcs, index)
                result +=  bar_mu * s(bcs, index)
                return result
            p_SSI.source = source_coef
            
            A = p_bform.assembly()
            b = p_lform.assembly()
            return A , b
        
        return update_us, update_ps, update_velocity, update_gauge, update_pressure












# class AllenCahn(FEMBase):
    # def method(phispace, init_phase, q):
    #     mesh = phispace.mesh
    #     init_mass = mesh.integral(init_phase)

    #     def update_ac(u_n, phi_n, t, dt, cm, cd, cc, source, mv: None):
    #         # 移动网格行为产生的速度场 mv 可选
    #         if mv is None:
    #             def mv(bcs):
    #                 NC = mesh.number_of_cells()
    #                 GD = mesh.geo_dimension()
    #                 shape = (NC, bcs.shape[0], GD)
    #                 return bm.zeros(shape, dtype=bm.float64)
                
    #         bform = BilinearForm(phispace)
    #         lform = LinearForm(phispace)
            
    #         SMI = ScalarMassIntegrator(coef=1.0+dt*(cm), q=q)
    #         SDI = ScalarDiffusionIntegrator(coef=dt*cd, q=q)
    #         SCI = ScalarConvectionIntegrator(q=q)
    #         SSI = SourceIntegrator(q=q)
            
    #         bform.add_integrator(SMI, SDI, SCI)
    #         lform.add_integrator(SSI)
            
    #         @barycentric
    #         def convection_coef(bcs, index):
    #             result = dt * (u_n(bcs, index) - mv(bcs, index))
    #             return result

    #         source = source(phi_n, dt, t)
            
    #         SCI.coef = convection_coef
    #         SSI.source = source
    #         A = bform.assembly()
    #         b = lform.assembly()

    #         def lagrange_multiplier(A, b):
    #             LagLinearForm = LinearForm(phispace)
    #             Lag_SSI = SourceIntegrator(source=1, q=q)
    #             LagLinearForm.add_integrator(Lag_SSI)
    #             LagA = LagLinearForm.assembly()
    #             A0 = -dt * cd * bm.ones(phispace.number_of_global_dofs())
    #             A0 = COOTensor(bm.array([bm.arange(len(A0), dtype=bm.int32), 
    #                                     bm.zeros(len(A0), dtype=bm.int32)]), A0, spshape=(len(A0), 1))
    #             A1 = COOTensor(bm.array([bm.zeros(len(LagA), dtype=bm.int32),
    #                                     bm.arange(len(LagA), dtype=bm.int32)]), LagA,
    #                                     spshape=(1, len(LagA)))
    #             b0 = bm.array([init_mass], dtype=bm.float64)
    #             A_block = BlockForm([[A, A0], [A1, None]])
    #             A_block = A_block.assembly_sparse_matrix(format='csr')
    #             b_block = bm.concat([b, b0], axis=0)
    #             return A_block, b_block
            
    #         A_block, b_block = lagrange_multiplier(A, b)
    #         return A_block, b_block
        
    #     return update_ac


# class GaugeUzawaNS(FEMBase):
    
#     def method(uspace, pspace, 
#                boundary, 
#                is_boundary, 
#                q = 3, 
#                apply_bc = None):  

#         mesh = uspace.mesh
#         us_bform = BilinearForm(uspace)
#         ps_bform = BilinearForm(pspace)
#         u_bform = BilinearForm(uspace)
#         s_bform = BilinearForm(pspace)
#         p_bform = BilinearForm(pspace)
        
#         us_lform = LinearForm(uspace)
#         ps_lform = LinearForm(pspace)
#         u_lform = LinearForm(uspace)
#         s_lform = LinearForm(pspace)
#         p_lform = LinearForm(pspace)
        
#         us_SMI = ScalarMassIntegrator(q=q)
#         us_SMI_phiphi = ScalarMassIntegrator(q=q)
#         us_VWI = ViscousWorkIntegrator(q=q)
#         us_SCI = ScalarConvectionIntegrator(q=q)
#         us_SSI = SourceIntegrator(q=q)
        
#         ps_SDI = ScalarDiffusionIntegrator(q=q)
#         ps_SSI = SourceIntegrator(q=q)
        
#         u_SMI = ScalarMassIntegrator(coef=1.0 , q=q)
#         u_SSI = SourceIntegrator(q=q)
        
#         s_SMI = ScalarMassIntegrator(coef=1.0 , q=q)
#         s_SSI = SourceIntegrator(q=q)
        
#         p_SMI = ScalarMassIntegrator(coef=1.0+1e-8 , q=q)
#         p_SSI = SourceIntegrator(q=q)
        
#         us_bform.add_integrator(us_SMI, us_VWI, us_SCI , us_SMI_phiphi)
#         us_lform.add_integrator(us_SSI)
#         ps_bform.add_integrator(ps_SDI)
#         ps_lform.add_integrator(ps_SSI)
#         u_bform.add_integrator(u_SMI)
#         u_lform.add_integrator(u_SSI)
#         s_bform.add_integrator(s_SMI)
#         s_lform.add_integrator(s_SSI)
#         p_bform.add_integrator(p_SMI)
#         p_lform.add_integrator(p_SSI)
        
#         # bc = DirichletBC(uspace)
        
#         def update_us(phi_n, phi, t, dt, u_n, s_n, cm, cphi, cv, cc, us_source, mv = None, apply_bc = apply_bc):
#             # 移动网格行为产生的速度场 mv 可选
#             if mv is None:
#                 def mv(bcs):
#                     NC = mesh.number_of_cells()
#                     GD = mesh.geo_dimension()
#                     shape = (NC, bcs.shape[0], GD)
#                     return bm.zeros(shape, dtype=bm.float64)
            
#             cv = cv(phi)
#             cm = cm(phi)
#             cc = cc(phi)

#             # mu = viscosity(phi)
#             # rho = density(phi)
#             # rho_n = density(phi_n)

#             @barycentric
#             def mass_coef(bcs,index):
#                 uh0_val = u_n(bcs, index)
#                 guh0_val = u_n.grad_value(bcs, index)
#                 rho1_val = cm(bcs, index)
#                 grho1_val = cm.grad_value(bcs, index)
#                 div_u_rho = bm.einsum('cqii,cq->cq', guh0_val, rho1_val) 
#                 div_u_rho += bm.einsum('cqd,cqd->cq', grho1_val, uh0_val) 
#                 result0 = rho1_val + 0.5 * dt * div_u_rho
                
#                 # 添加移动网格项
#                 result1 = 0.5*dt*bm.einsum('cqi,cqi->cq', grho1_val, mv(bcs, index))
#                 result = result0 - result1
#                 return result
            
#             @barycentric
#             def phiphi_mass_coef(bcs, index):
#                 gphi_n = phi_n.grad_value(bcs, index)
#                 gphi_gphi = bm.einsum('cqi,cqj->cqij', gphi_n, gphi_n)
#                 result = (cphi * dt) * gphi_gphi
#                 return result
            
#             @barycentric
#             def convection_coef(bcs, index):
#                 uh_n_val = u_n(bcs, index)
#                 rho1_val = cc(bcs, index)
#                 result = rho1_val[...,None] * (uh_n_val - mv(bcs, index))
#                 return dt * result
            
#             source = us_source(phi_n, phi, u_n, s_n, dt, t, mv)
            
#             us_SMI.coef = mass_coef
#             us_SMI_phiphi.coef = phiphi_mass_coef
#             us_VWI.coef = dt * cv
#             us_SCI.coef = convection_coef
#             us_SSI.source = source
                        
#             A = us_bform.assembly()
#             b = us_lform.assembly()
            
#             if apply_bc is None:
#                 bc = DirichletBC(uspace)
#                 bc.gd = lambda p : boundary(p, t)
#                 A , b = bc.apply(A, b)
#             return A , b
        
#         def update_ps(phi, us, cd):
#             cd = cd(phi)

#             @barycentric
#             def diffusion_coef(bcs, index):
#                 result = 1 / cd(bcs, index)
#                 return result
            
#             @barycentric
#             def source_coef(bcs, index):
#                 uh_grad_val = us.grad_value(bcs, index)
#                 div_u = bm.einsum('cqii->cq', uh_grad_val)
#                 return div_u
            
#             ps_SDI.coef = diffusion_coef
#             ps_SSI.source = source_coef
            
#             A = ps_bform.assembly()
#             b = ps_lform.assembly()
#             return A , b
        
#         def update_velocity(phi, us, ps, cl):
#             cl = cl(phi)
            
#             @barycentric
#             def source_coef(bcs, index):
#                 result = us(bcs, index)
#                 result += (1/cl(bcs, index)[..., None] )* ps.grad_value(bcs, index)
#                 return result
            
#             u_SSI.source = source_coef
            
#             A = u_bform.assembly()
#             b = u_lform.assembly()
#             return A , b
        
#         def update_gauge(s_n, us):
#             @barycentric
#             def source_coef(bcs,index):
#                 result = s_n(bcs,index) - bm.einsum('cqii->cq', us.grad_value(bcs,index))
#                 return result
#             s_SSI.source = source_coef

#             A = s_bform.assembly()
#             b = s_lform.assembly()
#             return A , b
        
#         def update_pressure(s, ps, dt, cl):
#             @barycentric
#             def source_coef(bcs, index):
#                 result = -1/dt * ps(bcs, index)
#                 result +=  cl * s(bcs, index)
#                 return result
#             p_SSI.source = source_coef
            
#             A = p_bform.assembly()
#             b = p_lform.assembly()
#             return A , b
        
#         return update_us, update_ps, update_velocity, update_gauge, update_pressure
