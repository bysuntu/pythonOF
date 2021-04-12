import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from ofDataStructure import cellData
import sys

class ofMatrix:
    def __init__(self, field, dim):
        self.nInternalFaces = field.nInternalFaces
        self.nCells = field.nCells
        self.boundary = field.boundary
        self.neighbour = field.neighbour
        self.owner = field.owner
        self.dim = dim
        self.refPressure = False
        self.construction()
        
    def construction(self):
        # Upper
        self.upperAddr = np.zeros((self.nInternalFaces, 1), dtype = np.int)
        self.upperV = np.zeros((self.nInternalFaces, 1), dtype = np.float)
        # Lower
        self.lowerAddr = np.copy(self.upperAddr)
        self.lowerV = np.copy(self.upperV)
        # Diag
        self.diagV = np.zeros((self.nCells, 1), dtype = np.float)
        self.source = np.copy(self.diagV)

        # Define values
        self.upperAddr[:, 0] = self.neighbour
        self.lowerAddr[:, 0] = self.owner[:self.nInternalFaces]

        self.internalCoeffs = {}
        self.boundaryCoeffs = {}
        for bc in self.boundary:
            self.internalCoeffs[bc[0]] = np.zeros((bc[-2], self.dim), dtype = np.float)
            self.boundaryCoeffs[bc[0]] = np.copy(self.internalCoeffs[bc[0]])
            
            
    def addToInternalField(self, cmptAv, diagV):
        for bc in self.boundary:
            name_, type_, range_, start_ = bc
            diagV[self.field.owner[start_:start_ + range_]] += cmptAv[name_]
            
            
    def addCmptAvBoundaryDiag(self, diagV):
        cmptAv = {}
        for bc in self.boundary:
            name_ = bc[0]
            cmptAv[name_] = np.mean(self.internalCoeffs[name_], axis = 1, keepdims = True)
        self.addToInternalField(cmptAv, diagV)
            
            
    def A(self):
        cA = np.copy(self.diagV)
        self.addCmptAvBoundaryDiag(cA)
        cA /= self.V
        bA = {}
        for bc in self.boundary:
            name_, type_, range_, start_ = bc
            if type_ == 'empty':
                bA[name_] = {'type': 'empty', 'value': cA[self.owner[start_:start_ + range_]]}
            else:
                bA[name_] = {'type': 'extrapolatedCalculated', 'value': cA[self.owner[start_:start_ + range_]]}
        return cellData(cA, bA)
            
            
    def lduH(self, U):
        h = np.zeros((self.nCells, 3), dtype = np.float)
        uPtr = self.upperAddr
        lPtr = self.lowerAddr
        psiPtr = U

        for i in range(self.nInternalFaces):
            h[uPtr[i]] -= self.lowerV[i] * psiPtr[lPtr[i]]
            h[lPtr[i]] -= self.upperV[i] * psiPtr[uPtr[i]]
        return h
            
            
    def addBoundarySource(self, coupled = False):
        source = np.zeros((self.nCells, 3), dtype = np.float)
        pbc = {}
        for bc in self.boundary:
            name_, type_, range_, start_ = bc
            pbc[name_] = self.boundaryCoeffs[name_]
        self.addToInternalField(pbc, source)
        return source
            
            
    def H(self):
        U = self.field.U.c
        cH = np.zeros((self.nCells, self.dim))
        for k in range(self.dim):
            for bc in self.boundary:
                name_, type_, range_, start_ = bc
                cH[self.owner[start_:start_ + range_], k] = self.internalCoeffs[name_][:, k] * -1
                cH[self.owner[start_:start_ + range_], k] += np.mean(self.internalCoeffs[name_], axis = 1)
            cH[:, k] *= U[:, k]
        
        cH += self.lduH(U) + self.source
        cH += self.addBoundarySource()
        cH /= self.V
        
        bH = {}
        for bc in self.boundary:
            name_, type_, range_, start_ = bc
            if type_ != 'empty':
                value_ = cH[self.owner[start_:start_ + range_]]
                bH[name_] = {'type':'extrapolatedCalculated', 'value':value_}
            else:
                bH[name_] = {'type': 'empty', 'value': cH[self.owner[start_:start_ + range_]]}
        return cellData(cH, bH)
        

    def setReference(self, pRefCell, pRefValue):
        self.pRefCell = pRefCell
        self.pRefValue = pRefValue
        self.refPressure = True

        
    def solve(self, equation):
        nsMatrix, fvcSource = equation
        self.lowerV = nsMatrix.lowerV
        self.upperV = nsMatrix.upperV
        self.diagV = nsMatrix.diagV
        self.source = nsMatrix.source
        self.internalCoeffs = nsMatrix.internalCoeffs
        self.boundaryCoeffs = nsMatrix.boundaryCoeffs

        if self.refPressure:
            # For pressure reference
            self.source[self.pRefCell] += self.diagV[self.pRefCell] * self.pRefValue
            self.diagV[self.pRefCell] += self.diagV[self.pRefCell]

        
        nC = self.field.nCells
        d = np.arange(nC).reshape(nC, 1)
        u = self.upperAddr
        l = self.lowerAddr
        
        c = np.concatenate((u, d, l), axis = 0).flatten()
        r = np.concatenate((l, d, u), axis = 0).flatten()
        v = np.concatenate((self.upperV, self.diagV, self.lowerV), axis = 0).flatten()
        
        b = np.copy(self.source)
        for k in range(self.dim):
            diagV_ = np.copy(self.diagV)
            for bc in self.field.boundary:
                name_, type_, range_, start_ = bc
                # Insert InternalCoeffs
                diagV_[self.field.owner[start_:start_ + range_], 0] += self.internalCoeffs[name_][:, k]
                b[self.field.owner[start_:start_ + range_], k] += self.boundaryCoeffs[name_][:, k]

            v = np.concatenate((self.upperV, diagV_, self.lowerV), axis = 0).flatten()

        Mk = coo_matrix((v, (r, c)), shape = (nC, nC))
        b += fvcSource * self.field.V
        
        if self.name == 'UEqn':
            self.field.UOldTime.c = self.field.U.c
            self.field.U.c = spsolve(Mk.tocsr(), b)
        elif self.name == 'PEqn':
            self.field.P.c[:, 0] = spsolve(Mk.tocsr(), b)
        else:
            sys.exit('FVM for {} is not defined yet!')


    def faceH(self):
        l = self.lowerAddr[:, 0]
        u = self.upperAddr[:, 0]
        faceHpsi = np.zeros((self.field.nFaces, 1), dtype = np.float)
        nIF = self.field.nInternalFaces
        if self.name == 'PEqn':
            faceHpsi[:nIF] = self.upperV * self.field.P.c[u] 
            faceHpsi[:nIF] -= self.lowerV * self.field.P.c[l]
            return faceHpsi
        else:
            return


    def flux(self):
        fieldFlux = self.faceH()
        
        internalContrib = {}
        for bc in self.field.boundary:
            name_, type_, range_, start_ = bc
            internalCoeffs_ = self.internalCoeffs[name_]
            patchInternalField_ = self.field.P.c[self.field.owner[start_:start_ + range_]]
            internalContrib[name_] = {'name': name_, 'value':patchInternalField_ * internalCoeffs_}
            fieldFlux[start_:start_ + range_] = patchInternalField_ * internalCoeffs_
        
        if self.fluxRequired:
            fieldFlux += self.faceFluxCorrectionPtr
            
        return fieldFlux

    