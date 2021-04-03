import os
import numpy as np
from ofIO import readFieldScalar, readFieldVector
from ofMesh import ofMesh
from ofDataStructure import cellData


class ofField(ofMesh):
    def __init__(self, meshDir, timeDir, dt = 0):
        super().__init__(meshDir)
        self.dt = dt
        self.loadField(timeDir)
        self.updateField()

    def updateField(self):
        # Cell center values
        self.UOldTime = self.U.copy()
        self.cPOldTime = self.P.copy()
        # Flux values
        self.phiOldTime = np.copy(self.phi)


    def loadField(self, timeDir):
        # Load Velocity
        uFileName = os.path.join(timeDir, 'U')
        cU, bU = readFieldVector(uFileName, self.nCells, self.nInternalFaces)
        self.U = cellData(cU, bU)
        
        # Load Pressure
        pFileName = os.path.join(timeDir, 'p')
        cP, bP = readFieldScalar(pFileName, self.nCells, self.nInternalFaces)
        self.P = cellData(cP, bP)
        
        # Load Flux. If phi is absent, need to calculate it.
        try:
            phiFileName = os.path.join(timeDir, 'phi')
            self.phi = np.zeros((self.nFaces, 1), dtype = np.float)
            iPhi, bPhi = self.readFieldScalar(phiFileName)
            self.phi[:self.nIF] = iPhi
            for bc in self.boundary:
                name_, _, range_, start_ = bc
                try:
                    self.phi[start_:start_ + range_] = bPhi[name_]['value']
                except ValueError:
                    pass
        except AttributeError:
            self.phi = self.updatePhi(self.U)


    def updatePhi(self, U):
        cU, bU = U.c, U.b
        phi = np.zeros((self.nFaces, 1))
        # Internal
        self.dotInterpolation(cU, self.sF, phi[:self.nInternalFaces])
        # Boundary
        # boundaryPhi = sF
        for bc in self.boundary:
            name_, type_, range_, start_ = bc
            sF_ = self.sF[start_:start_ + range_]
            if bU[name_]['type'] == 'fixedValue' or bU[name_]['type'] == 'extrapolatedCalculated':
                u_ = bU[name_]['value']
                phi[start_:start_ + range_, 0] = np.sum(sF_ * u_, axis = 1)
            elif bU[name_]['type'] == 'empty':
                pass
            else:
                print('Unknown boundary condition', name_, bU[name_]['type'])
        return phi

    def dotInterpolation(self, cellValue, internalSurfaceVector = None, faceValue = None):
        owner = self.owner
        neighbour = self.neighbour
        nIF = self.nInternalFaces
        weights = self.weights
        
        sF = internalSurfaceVector
        vF = cellValue
        svF = (vF[owner[:nIF]] - vF[neighbour]) * weights[:nIF] + vF[neighbour]
        if faceValue is None:
            return svF
        if cellValue.shape[1] == 3: # Vector
            faceValue[:nIF] = np.sum(sF[:nIF] * svF, axis = 1, keepdims=True)
        elif cellValue.shape[1] == 9: # Tensor
            for k in range(3):
                faceValue[:nIF, k] = np.sum(sF[:nIF] * svF[:, k::3], axis = 1)
        elif cellValue.shape[1] == 1: # Scalar
            faceValue[:nIF] = svF
        else:
            raise TypeError("The input vector shape is not allowed")
