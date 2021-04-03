import os
import numpy as np
from ofIO import readScalar, readVector, readList, readBC

ROOTVSMALL = 10e-12


class ofMesh:
    def __init__(self, meshDir, meshName = 'polyMesh'):
        self.loadMesh(meshDir, meshName)
        self.updateMesh()

    def loadMesh(self, meshDir, meshName):
        oFileName = os.path.join(meshDir, meshName, 'owner')
        self.owner = readScalar(oFileName)
        nFileName = os.path.join(meshDir, meshName, 'neighbour')
        self.neighbour = readScalar(nFileName)
        pFileName = os.path.join(meshDir, meshName, 'points')
        self.points = readVector(pFileName)
        fFileName = os.path.join(meshDir, meshName, 'faces')
        self.faces = readList(fFileName)
        bFileName = os.path.join(meshDir, meshName, 'boundary')
        self.boundary = readBC(bFileName)        

    def updateMesh(self):
        self.nCells = max(self.owner) + 1 # Number of Cells
        self.nFaces = len(self.faces)     # Number of faces
        self.nInternalFaces = len(self.neighbour) # Number of internal faces

        self._getFaceNeighbour()
        self._getSurface()
        self._getVolume()
        self._getWeight()
        self._getNonOrthDeltaCoeffs()
        self._getNonOrthCorrectionVectors()

    def _getNonOrthDeltaCoeffs(self):
        coeffs = np.zeros((self.nFaces, 1))
        nIF = self.nInternalFaces
        delta = self.cC[self.neighbour] - self.cC[self.owner[:nIF]]
        unitArea = self.sF / self.magSF
        deltaDunitArea = np.sum(delta * unitArea[:nIF], axis = 1)
        magDelta = np.linalg.norm(delta, axis = 1)
        coeffs[:nIF, 0] = 1. / np.max([deltaDunitArea, 0.05 * magDelta], axis = 0)

        # Boundary Patch
        for bc in self.boundary:
            _, type_, range_, start_ = bc
            if type_ == 'empty':
                continue

            fC_ = self.fC[start_:start_ + range_]
            cC_ = self.cC[self.owner[start_:start_ + range_]]
            delta_ = fC_ - cC_
            delta_ = np.sum(delta_ * unitArea[start_:start_ + range_], axis = 1, keepdims=True)
            delta_ = delta_ * unitArea[start_:start_ + range_]
            
            dDA = np.sum(delta_ * unitArea[start_:start_ + range_], axis = 1)
            mD = np.linalg.norm(delta_, axis = 1)
            coeffs[start_:start_ + range_, 0] = 1. / np.max([dDA, 0.05 * mD], axis = 0) 

        self.nonOrthDeltaCoeffs = coeffs


    def _getNonOrthCorrectionVectors(self):
        corrVecs = np.zeros((self.nFaces, 3))
        nIF = self.nInternalFaces
        unitArea = self.sF / self.magSF
        delta = self.cC[self.neighbour] - self.cC[self.owner[:nIF]]
        
        corrVecs[:nIF] = unitArea[:nIF] - delta * self.nonOrthDeltaCoeffs[:nIF]
        self.nonOrthCorrectionVectors = corrVecs


    def _getWeight(self):
        weights = np.zeros(self.nFaces, dtype = np.float32)
        for k in range(len(self.neighbour)):
            own_ = np.abs(np.dot(self.sF[k], self.fC[k] - self.cC[self.owner[k]]))
            nei_ = np.abs(np.dot(self.sF[k], self.cC[self.neighbour[k]] - self.fC[k]))
            weights[k] = nei_ / (own_ + nei_)
            
        for bc_ in self.boundary:
            _, type_, num_, start_ = bc_
            num_ = int(num_)
            start_ = int(start_)
            for k in range(start_, start_ + num_):
                if type_ == 'empty':
                    weights[k] = 0
                else:
                    weights[k] = 1.
        self.weights = weights.reshape(-1, 1)


    def _getVolume(self):
        volumes = np.zeros((self.nCells, 1))
        cellCentroids = np.zeros((self.nCells, 3))
        
        for k, fs_ in enumerate(self.fN):
            cC_ = np.mean(self.fC[fs_], axis = 0) # Initial guess
            cellCentroids[k] = cC_
            cellVolume = 0
            sum_ = np.zeros(3)
            for fI in fs_: # fI
                fC_ = self.fC[fI] # Face center
                _, p0_ = self.faces[fI]
                p1_ = p0_[1:] + p0_[:1]
                pts0_ = self.points[p0_]
                pts1_ = self.points[p1_]
                
                tC_ = (pts0_ + pts1_ + fC_ + cC_) / 4 # Tetra centroid
                tV_ = np.dot(np.cross(pts0_ - fC_, pts1_ - fC_), cC_ - fC_) * 1 / 6
                tV_ = np.abs(np.expand_dims(tV_, axis = 1))

                sum_ += np.sum(tV_ * tC_, axis = 0)
                cellVolume += np.sum(tV_)
            cC_ = sum_ / cellVolume # Update
            volumes[k, 0] = cellVolume
            cellCentroids[k] = cC_

        self.V = volumes
        self.cC = cellCentroids


    def _getFaceNeighbour(self):
        faceNeighbours = [[] for _ in range(self.nCells)]

        for k, c_ in enumerate(self.owner):
            faceNeighbours[c_].append(k)
        for k, c_ in enumerate(self.neighbour):
            faceNeighbours[c_].append(k)
        self.fN = faceNeighbours


    def _getSurface(self):
        faces = self.faces
        points = self.points
    
        self.sF = np.zeros((len(faces), 3), dtype = np.float32)
        self.fC = np.zeros((len(faces), 3), dtype = np.float32)

        for i, f_ in enumerate(faces):
            nPts_, Pts_ = f_
            if nPts_ == 3:
                fCtr_ = (f_[0] + f_[1] + f_[2]) / 3
                fArea_ = np.cross(f_[1] - f_[0], f_[2] - f_[0]) * 0.5
            else:
                fCenter_ = np.mean(points[Pts_], axis = 0)
    
                p0 = points[Pts_]
                p1 = points[Pts_[1:] + [Pts_[0]]]

                aVectors_ = np.cross(p1 - p0, fCenter_ - p0) * 0.5

                fArea_ = np.sum(aVectors_, axis = 0)
                area_ = np.linalg.norm(aVectors_, axis = 1, keepdims=True)
                fCtr_ = np.sum((p0 + p1 + fCenter_) * area_, axis = 0) / np.sum(area_) * 1 / 3

            if np.sum(area_) < ROOTVSMALL:
                fCtr_ = fCenter_
                fArea_ = 0
            
            self.sF[i] = fArea_
            self.fC[i] = fCtr_

        self.magSF = np.linalg.norm(self.sF, axis = 1, keepdims=True)


