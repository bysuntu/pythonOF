import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import sys
from ofMatrix import ofMatrix
from ofDataStructure import cellData, matrixData

ROOTVSMALL = 10e-12

def nbGrad(own, nei, sF, sV, igGrad, fIds):
    nIF = len(nei)
    if igGrad.shape[1] == 3: # Vector
        for i in fIds:
            igGrad[own[i]] += sF[i] * sV[i]
            if i < nIF:
                igGrad[nei[i]] -= sF[i] * sV[i]
    elif igGrad.shape[1] == 9: # Tensor
        for i in fIds:
            for k in range(3):
                igGrad[own[i], k::3] += sF[i] * sV[i, k]
                if i < nIF:
                    igGrad[nei[i], k::3] -= sF[i] * sV[i, k]
                    

def gauss_Seidel(M, b, x, tol = 10e-5, num = 10):
    n = len(b)
    for i in range(num):
        for r in range(n):
            x[r] *= 0
            x[r] = (b[r] - np.matmul(M[r], x)) / M[r, r]
        err = np.mean(np.linalg.norm(b - np.matmul(M, x), axis = 1))
        if err < tol:
            return


def nbSurfaceIntegrate(owner, neighbour, surfaceV, cellV):
    for i in range(len(neighbour)):
        cellV[owner[i]] += surfaceV[i]
        if i < len(neighbour):
            cellV[neighbour[i]] -= surfaceV[i]
    return
                    

def nbNegSumDiag(upperAddr, lowerAddr, upperV, lowerV, diagV):
    for i in range(len(upperAddr)):
        diagV[lowerAddr[i]] -= lowerV[i]
        diagV[upperAddr[i]] -= upperV[i]
    return


class FVM(ofMatrix): # Finite Volume Method
    def __init__(self, field, name): # filed: P, U. name = Type. dim = Dimension
        # Transfer variables from field to FVM such as velocity and pressure
        self.updateFieldData(field, name)
    

    def updateFieldData(self, field, name):
        # Type of this FVM equation
        nature = {'UEqn': [False, 3], 'PEqn': [True, 1]}
        self.fluxRequired, self.dim = nature[name]

        super().__init__(field, self.dim)

        # For non orthogonal grids, face correction is needed 
        self.faceFluxCorrectionPtr = None

        # Transfer data from field to FVM
        self.field = field
        self.name = name
        
        self.owner = field.owner
        self.neighbour = field.neighbour
        self.boundary = field.boundary

        self.nCells = field.nCells
        self.nFaces = field.nFaces
        self.nInternalFaces = field.nInternalFaces
        self.dt = field.dt
        self.V = field.V


    def dudt(self, scheme = 'Euler'): # Time
        rDeltaT = 1. / self.dt
        # Lower Upper Dia Source InternalCoeffs BoundaryCoeffs
        # Also, this serves as the initialization of matrix
        lowerV= 0
        upperV = 0
        diagV = rDeltaT * self.field.V
        source = rDeltaT * self.field.U.c * self.field.V

        for bc in self.field.boundary:
            name_, type_, range_, start_ = bc
            self.internalCoeffs[name_] *= 0
            self.boundaryCoeffs[name_] *= 0
        return matrixData(lowerV, upperV, diagV, source, self.internalCoeffs, self.boundaryCoeffs)


    def div(self, varStr, scheme = 'Gauss'): # Divergence
        if self.name == 'UEqn' and varStr == 'phi, U':
            nIF = self.field.nInternalFaces
            lowerV = -self.field.weights[:nIF] * self.field.phi[:nIF]
            upperV = self.field.phi[:nIF] - self.field.weights[:nIF] * self.field.phi[:nIF]
            diagV = self.diagV * 0
            nbNegSumDiag(self.upperAddr, self.lowerAddr, upperV, lowerV, diagV)

            internalCoeffs = {}
            boundaryCoeffs = {}
            for bc in self.field.boundary:
                name_, type_, range_, start_ = bc
                phi_ = self.field.phi[start_:start_ + range_]
                weights_ = self.field.weights[start_:start_ + range_]
                internalCoeffs[name_] = phi_ * weights_
                boundaryCoeffs[name_] = phi_ * weights_
            return matrixData(lowerV, upperV, diagV, 0 * self.source, internalCoeffs, boundaryCoeffs)
        elif self.name =='PEqn':
            phi = varStr
            cValue = np.zeros((self.field.nCells, 1))
            nbSurfaceIntegrate(self.field.owner, self.field.neighbour, 
                               phi, cValue)
            cValue /= self.field.V
            bValue = {}
            for bc in self.field.boundary:
                name_, type_, range_, start_ = bc
                if type_ == 'empty':
                    bValue[name_] = {'type':'empty'}
                else:
                    bValue[name_] = {'type':'extrapolatedCalculated',
                                     'value':cValue[self.field.owner[start_:start_ + range_]]}
            return cValue, bValue


    def laplacian(self, gamma, varStr, limitCoeff, snGradSchemeCorr = True, meshFluxRequired = False):
        # meshFlus is required for dynamic mesh only
        # snGradSchemeCorr is used for non-orthognal correction

        gammaMagSf = self.getGammaMagSf(gamma)
        # Uncorrected Part
        uncorrectedPart = self.laplacianUncorrected(gammaMagSf, varStr)

        # Correction Part added to source term
        ivf = np.zeros((self.field.nCells, self.dim), dtype = np.float)
        if snGradSchemeCorr and not meshFluxRequired:
            nuGradU = gammaMagSf * self.limitedSnGrad(varStr, limitCoeff)
            if self.fluxRequired:
                self.faceFluxCorrectionPtr = nuGradU
            nbSurfaceIntegrate(self.owner, self.neighbour, nuGradU, ivf)
            vsc = self.field.V
            ivf /= vsc
            correctedPart = -ivf * self.field.V
        else:
            sys.exit("This is not implemented!")
        
        # Combine uncorrected and corrected parts
        lowerV, upperV, diagV, source, internalCoeffs, boundaryCoeffs = uncorrectedPart
        source = correctedPart
        return matrixData(lowerV, upperV, diagV, source, internalCoeffs, boundaryCoeffs)


    def laplacianUncorrected(self, gammaMagSf, varStr):
        nature = {'U': self.field.U.b, 'P': self.field.P.b}
        bU = nature[varStr]

        # Update matrix
        nIF = self.field.nInternalFaces
        upperV = self.field.nonOrthDeltaCoeffs[:nIF] * gammaMagSf[:nIF]
        lowerV = upperV

        diagV = self.diagV * 0
        nbNegSumDiag(self.upperAddr, self.lowerAddr, upperV, lowerV, diagV)

        internalCoeffs, boundaryCoeffs = {}, {}
        for bc in self.field.boundary:
            name_, type_, range_, start_ = bc
            pGamma = gammaMagSf[start_:start_ + range_]
            
            if bU[name_]['type'] == 'fixedValue':
                internalCoeffs[name_] = -1 * np.ones(self.dim) * pGamma * self.field.nonOrthDeltaCoeffs[start_:start_ + range_]
                boundaryCoeffs[name_] = -1 * pGamma * self.field.nonOrthDeltaCoeffs[start_:start_ + range_] * bU[name_]['value']
            else:
                internalCoeffs[name_] = self.internalCoeffs[name_] * 0
                boundaryCoeffs[name_] = self.boundaryCoeffs[name_] * 0

        return lowerV, upperV, diagV, 0, internalCoeffs, boundaryCoeffs


    def getGammaMagSf(self, gamma):
        if np.isscalar(gamma):
            gamma = np.ones((self.field.nFaces, 1)) * gamma
            for bc in self.field.boundary:
                name_, type_, range_, start_ = bc
                if type_ == 'empty':
                    gamma[start_:start_ + range_] = 0
        else:
            cGamma, bGamma = gamma.c, gamma.b
            gamma = np.zeros((self.field.nFaces, 1))
            self.field.dotInterpolation(cGamma, self.field.sF, gamma)
            for bc in self.field.boundary:
                name_, type_, range_, start_ = bc
                try:
                    gamma[start_:start_ + range_] = bGamma[name_]['value']
                except KeyError:
                    pass
        gammaMagSf = gamma * self.field.magSF
        return gammaMagSf
        

    def grad(self, varStr, scheme = 'Gauss'):
            
        nature = {}
        nature['U'] = [self.field.U.c, self.field.U.b, np.zeros((self.nCells, 9))]
        nature['P'] = [self.field.P.c, self.field.P.b, np.zeros((self.nCells, 3))]
        
        cValue, bValue, cGrad = nature[varStr]
        
        # Surface values
        # Initialization
        svF = np.zeros((self.nFaces, cValue.shape[1]), dtype = np.float32)
        # Internal values 
        svF[:self.nInternalFaces] = self.field.dotInterpolation(cValue)
        # Boundary values
        for bc in self.boundary:
            name_, type_, range_, start_ = bc
            if bValue[name_]['type'] == 'empty':
                continue
            elif bValue[name_]['type'] == 'fixedValue':
                svF[start_:start_ + range_] = bValue[name_]['value']
            elif bValue[name_]['type'] == 'zeroGradient':
                svF[start_:start_ + range_] = cValue[self.owner[start_:start_ + range_]]
            else:
                raise TypeError("{} for {} is not defined!".format(name_, bValue[name_]['type']))
        
        # Calculate gradient
        nbGrad(self.owner, self.neighbour, self.field.sF, svF, cGrad, 
               np.arange(self.nFaces))
        
        cGrad /= self.V
        
        # Boundary Get from boundary conditions
        bGrad = {}
        for bc in self.boundary:
            name_, type_, range_, start_ = bc
            if bValue[name_]['type'] == 'empty':
                bGrad[name_] = {'type': 'empty'}
            elif bValue[name_]['type'] == 'fixedValue':
                bGrad[name_] = {'type': 'extrapolatedCalculated', 'value':cGrad[self.owner[start_:start_ + range_]]}
            elif bValue[name_]['type'] == 'zeroGradient':
                nF = np.copy(self.field.sF[start_:start_ + range_])
                nF /= np.linalg.norm(nF, axis = 1, keepdims = True)
                bInternalV = cGrad[self.owner[start_:start_ + range_]]
                bGrad[name_] = {'type': 'extrapolatedCalculated', 'value': bInternalV - nF * np.sum(bInternalV * nF, axis = 1, keepdims=True)}
            else:
                raise TypeError("{} is not defined!".format(bValue[name_]['type']))
            
        return cGrad, bGrad


    def correctedSnGrad(self, cStr, scheme = 'Gauss'):
        cGrad, bGrad = self.grad(cStr, scheme)
        # Correction is applied to internal surfaces only

        nature = {'U': 3, 'P': 1}
        dim = nature[cStr]
        
        svgDotCorrV = np.zeros((self.nFaces, dim), dtype = np.float)
        self.field.dotInterpolation(cGrad, self.field.nonOrthCorrectionVectors, svgDotCorrV)
        return svgDotCorrV
        
        
    def snGrad(self, cStr, scheme = 'Gauss'):
        nature = {}
        nature['U'] = [self.field.U.c, self.field.U.b, np.zeros((self.nFaces, 3))]
        nature['P'] = [self.field.P.c, self.field.P.b, np.zeros((self.nFaces, 1))]
        cValue, bValue, ssf = nature[cStr]
            
        # Internal
        dV = cValue[self.neighbour] - cValue[self.owner[:self.nInternalFaces]]
        ssf[:self.nInternalFaces] = self.field.nonOrthDeltaCoeffs[:self.nInternalFaces] * dV
        
        for bc in self.boundary:
            name_, type_, range_, start_ = bc
            if bValue[name_]['type'] == 'empty':
                continue
            elif bValue[name_]['type'] == 'fixedValue':
                curRange = np.arange(start_, start_ + range_)
                dV = bValue[name_]['value'] - cValue[self.owner[curRange]]
                ssf[curRange] = self.field.nonOrthDeltaCoeffs[curRange] * dV
            elif bValue[name_]['type'] == 'zeroGradient':
                curRange = np.arange(start_, start_ + range_)
                dV = 0
                ssf[curRange] = self.field.nonOrthDeltaCoeffs[curRange] * dV
            else:
                raise TypeError("{} is unknown!".format(bValue[name_]['type']))
        return ssf
        
        
    def limitedSnGrad(self, cStr, limitCoeff_):
        corr = self.correctedSnGrad(cStr)
        magCorr = np.linalg.norm(corr, axis = 1, keepdims=True)
        snGrad = self.snGrad(cStr)
        magSnGrad = np.linalg.norm(snGrad, axis = 1, keepdims=True)
        limiter = np.ones((self.nFaces, 2))
        limiter[:, :1] = limitCoeff_ * magSnGrad / ((1 - limitCoeff_) * magCorr + ROOTVSMALL)
        limiter = np.min(limiter, axis = 1, keepdims = True)
        return limiter * corr
        
        
class FVC(FVM):
    def __init__(self, field, name):
        super().__init__(field, name)

    def grad(self, varStr, scheme = 'Gauss'):
        cG, bG = super().grad(varStr, scheme)
        return cG

    def div(self, varStr, scheme = 'Gauss'): # Divergence
        cD, bD = super().div(varStr, scheme)
        return cD