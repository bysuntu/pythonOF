from ofFVM import FVM, FVC
import numpy as np

ROOTVSMALL = 10e-12
# Reference values
pRefCell = 0
pRefValue = 0

class PISO:
    def __init__(self, NS, nCorrectors, nNonOrthogonalCorrectors, limitCoeff):
        self.limitCoeff = limitCoeff
        self.nCorrectors = nCorrectors
        self.nNonOrthogonalCorrectors = nNonOrthogonalCorrectors
        self.field = NS.field
        self.pisoLoop(NS)

    def pisoLoop(self, NS):
        for _ in range(self.nCorrectors):
            A = NS.A()
            H = NS.H()

            rAU = A.inv()
            HbyA = rAU * H

            self.constrainHbyA(HbyA.b, NS.field.U.b, NS.field.P.b)
            phiHbyA = self.calPhiHbyA(NS.field, A, rAU, HbyA)

            self.adjustPhi()
            self.constrainPressure()

            for k in range(self.nNonOrthogonalCorrectors + 1):
                pGrad = self.nonOrthogonalLoop(rAU, phiHbyA, k == self.nNonOrthogonalCorrectors)

            pCorr = rAU.c * pGrad
            self.field.U.c = HbyA.c - pCorr


    def nonOrthogonalLoop(self, rAU, phiHbyA, finalNonOrthogonal):
        print("NonOrthogonal Iteration")

        fvm = FVM(self.field, 'PEqn')
        fvc = FVC(self.field, 'PEqn')
        fvm.setReference(pRefCell, pRefValue)

        fvm.solve(fvm.laplacian(rAU, 'P', self.limitCoeff) == fvc.div(phiHbyA))

        if finalNonOrthogonal:
            print('NonOrthogonal Final')
            fvm.field.phi = phiHbyA - fvm.flux()
            return fvc.grad('P')
        else:
            return None

    def adjustPhi(self):
        pass


    def constrainPressure(self):
        pass


    def constrainHbyA(self, HbyA, U, P):
        # when velocity boundary condistion is not assignable and
        # fixedFluxExtrapolatedPressureFvPatchScalarField
        # boundary conditions of HbyA are the same as those of U

        for bc in U:
            if U[bc]['type'] == 'empty':
                HbyA[bc]['type'] = 'empty'
                HbyA[bc]['value'] = U[bc]['value']
            else:
                HbyA[bc]['type'] = 'extrapolatedCalculated'
                HbyA[bc]['value'] = U[bc]['value']


    def calPhiHbyA(self, field, A, rAU, HbyA):
        flux = field.updatePhi(HbyA)
        surfaceRAU = np.zeros((field.nFaces, 1))
        # Internal
        surfaceRAU[:field.nInternalFaces] = field.dotInterpolation(rAU.c, field.sF)
        # Boundary
        for bc in field.boundary:
            name_, type_, range_, start_ = bc
            surfaceRAU[start_:start_ + range_] = rAU.b[name_]['value']

        ddtCorr = self.fvcDdtPhiCorr(field)
        return flux + surfaceRAU * ddtCorr
        

    def fvcDdtPhiCorr(self, field, scheme = 'Euler'):
        rDeltaT = 1. / field.dt
        phi = np.zeros((field.nFaces, 1))
        field.dotInterpolation(field.UOldTime.c, field.sF, phi)
        phiCorr = field.phiOldTime - phi

        # Flux normalized forumulation
        ddtCouplingCoeff = np.ones((field.nFaces, 2), dtype = np.float)
        ddtCouplingCoeff[:, 0] = np.linalg.norm(phiCorr, axis = 1)
        ddtCouplingCoeff[:, :1] *= field.dt * field.nonOrthDeltaCoeffs / (field.magSF + ROOTVSMALL)
        ddtCouplingCoeff = 1. - np.min(ddtCouplingCoeff, axis = 1, keepdims = True)

        for bc in field.boundary:
            name_, type_, range_, start_ = bc
            if field.U.b[name_]['type'] == 'fixedValue':
                ddtCouplingCoeff[start_:start_ + range_] = 0
            elif field.U.b[name_]['type'] == 'empty':
                print('Check')
            else:
                print('{} is unknown'.format(name_))
        return ddtCouplingCoeff * rDeltaT * phiCorr


