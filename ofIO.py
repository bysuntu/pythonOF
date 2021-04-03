import os
import numpy as np
ROOTVSMALL = 10e-12


def readScalar(filename):
    with open(filename, 'r') as f:
        off = True
        lines = f.readlines()
        ftNum = 0
        ftList = []
        for line in lines:
            if line.find('//') == 0:
                off = not off
            if off == False:
                cur_ = line.replace('\n', '').replace(' ', '')
                if cur_.isdigit() and ftNum == 0:
                    ftNum = max(int(cur_), ftNum)
                    continue
                if cur_.isdigit():
                    ftList.append(int(cur_))
        return np.array(ftList)

def readVector(filename):
    with open(filename, 'r') as f:
        off = True
        lines = f.readlines()
        ptsNum = 0
        ptsList = []
        for line in lines:
            if line.find('//') == 0:
                off = not off
            if off == False:
                cur_ = line.replace('\n', '')
                if cur_.isdigit():
                    ptsNum = int(cur_)
                if line.find('(') == 0 and line.find(')') > 0:
                    xyz_ = [float(n) for n in line[1:line.find(')')].split(' ')]
                    ptsList.append(xyz_)
            if len(ptsList) == ptsNum and ptsNum > 0:
                break
        return np.array(ptsList)

def readList(filename):
    with open(filename, 'r') as f:
        off = True
        lines = f.readlines()
        ftNum = 0
        ftList = []
        for line in lines:
            if line.find('//') == 0:
                off = not off
            if off == False:
                cur_ = line.replace('\n', '')
                if cur_.isdigit():
                    ftNum = int(cur_)
                if line.find('(') > 0 and line.find(')') > 0:
                    n_ = int(line[:line.find('(')])
                    xyz_ = [int(n) for n in line[line.find('(')+1:line.find(')')].split(' ')]
                    ftList.append([n_, xyz_])
            if len(ftList) == ftNum and ftNum > 0:
                break
        return ftList

def readBC(filename):
    with open(filename, 'r') as f:
        off = True
        lines = f.readlines()
        bcNum = 0
        boundary = []
        for line in lines:
            if line.find('//') == 0:
                off = not off
            if off == False:
                cur_ = line.replace('\n', '')
                if cur_.isdigit():
                    bcNum = int(cur_)
            if off == False and line.find('\n') > 4 and line.find(';') < 0 and line.find('{') < 0 and line.find('}') < 0:
                each_ = [line[4:line.find('\n')]]
            if off == False and line.find('type') == 8:
                each_.append(line.split(' ')[-1][:-2])
            if off == False and line.find('nFaces') == 8:
                each_.append(int(line.split(' ')[-1][:-2]))
            if off == False and line.find('startFace') == 8:
                each_.append(int(line.split(' ')[-1][:-2]))
            if off == False and line.find('}') == 4:
                boundary.append(each_)
            if off == False and line.find(')') == 0:
                off = not off
        assert len(boundary) == bcNum
        return boundary

def readFieldVector(uFileName, nCells, nInternalFaces):
    with open(uFileName, 'r') as f:
        lines = f.readlines()
        readInternal = False
        readBoundary = False
        readList = False
        cI = 0
        U = np.zeros((nCells, 3))
        BC = {}
        for line in lines:
            if line.find('internalField') == 0:
                if line.split(' ')[1] == 'uniform':
                    digits_ = line[line.find('(') + 1: line.find(')')]
                    vector_ = np.array([float(k) for k in digits_.split(' ')])
                    U = np.tile(vector_, (nCells, 1))
                else: 
                    readInternal = True
                    continue
            if readInternal and line.find('(') == 0 and line.find(')') > 0:
                line = line.replace('\t', ' ')
                U[cI] = [float(k) for k in line[1:-2].split(' ')]
                cI += 1
            if line.find(';') == 0:
                readInternal = False
            if line.find('boundaryField') == 0:
                readBoundary = True

            if readBoundary and readList == False and line.find('{') < 0 and line.find('}') < 0 and line.find(';') < 0 and len(line) > 4 and line.find('uniform') < 0:
                name_ = line.replace('\t', ' ').replace(' ', '')[:-1]
                value_ = []
            if readBoundary and line.find('type') > 0 and line.find(';') > 0:
                type_ = line.replace('\t', ' ')[:-2].split(' ')[-1]
            if readBoundary and line.find('value') > 0 and line.find(';') > 0 and line.find('uniform') > 0 and line.find('nonuniform') < 0:
                value_ = line[line.find('(') + 1:line.find(')')].split(' ')
                value_ = np.array([float(k) for k in value_])
            if readBoundary and line.find('value') > 0 and line.find('nonuniform') > 0:
                readList = True
                value_ = []
            if readBoundary and readList and line.find('(') > 0 and line.find(')') > 0:
                curV_ = line[line.find('(') + 1:line.rfind(')')]
                try:
                    value_.append(np.array([float(k) for k in curV_.split(' ')]))
                except ValueError:
                    curV_ = curV_.replace('(', '').replace(')', '').split(' ')
                    value_ = np.array([float(k) for k in curV_]).reshape(-1, 3).tolist()
                    
            if readBoundary and readList and line.find(';') > 0:
                readList = False
            if readBoundary and line.find('}') > 0:
                BC[name_] = {'type': type_, 'value': value_}
            if readBoundary and line.find('}') == 0:
                readBoundary = False
                
        return U, BC

def readFieldScalar(pFileName, nCells, nInternalFaces):
    with open(pFileName, 'r') as f:
        lines = f.readlines()
        readInternal = 0
        readBoundary = False
        readList = False
        p = []
        BC = {}
        for line in lines:
            if line.find('internalField') == 0:
                if line.find('nonuniform') > 0:
                    readInternal = 1
                else:
                    p = np.ones(nCells) * float(line[:-2].split(' ')[-1])
                
            if readInternal == 1 and line.find('(') == 0:
                readInternal = 2
                continue
            if readInternal == 2 and line.find(')') == 0:
                readInternal = 0
            if readInternal == 2:
                p.append(float(line[:-1]))
            
            if line.find('boundaryField') == 0:
                readBoundary = True
                continue
                
            if readBoundary and line.find('{') < 0 and line.find('}') < 0 and line.find(';') < 0 and len(line) > 4 and line.find('uniform') < 0:
                name_ = line.replace('\t', ' ').replace(' ', '')[:-1]
            if readBoundary and line.find('type') > 0 and line.find(';') > 0:
                type_ = line[:-2].split(' ')[-1]
                
            if readBoundary and line.find('value') > 0 and line.find(';') > 0 and line.find('uniform') > 0 and line.find('nonuniform') < 0:
                value_ = line[line.find('(uniform') + 7:line.find(';')].split(' ')[-1]
                value_ = np.array([float(k) for k in value_])
            if readBoundary and line.find('value') > 0 and line.find('nonuniform') > 0:
                readList = True * (type_ == 'empty')
                value_ = []
            if readBoundary and readList and line.find('(') > 0 and line.find(')') > 0:
                print('Line', line)
                curV_ = line[line.find('(') + 1:line.rfind(')')]
                try:
                    value_.append(np.array([float(k) for k in curV_.split(' ')]))
                except ValueError:
                    curV_ = curV_.replace('(', '').replace(')', '').split(' ')
                    try:
                        value_ = np.array([float(k) for k in curV_]).reshape(-1, 3).tolist()
                    except ValueError:
                        pass
                    
            if readBoundary and line.find('}') > 0:
                try:
                    BC[name_] = {'type': type_, 'value': value_}
                except UnboundLocalError:
                    BC[name_] = {'type': type_}
                    
            if readBoundary and line.find('}') == 0:
                break
        try:
            return np.reshape(p, (nCells, 1)), BC
        except ValueError:
            return np.reshape(p, (nInternalFaces, 1)), BC

