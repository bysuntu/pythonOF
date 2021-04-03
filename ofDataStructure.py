class cellData:
    def __init__(self, cellData, boundaryData = None):
        self.c = cellData
        self.b = boundaryData
        pass

    def __add__(self, other):
        c = self.c + other.c
        b = {}
        for p in other.b.items():
            k, v = p
            b[k] = {'type': v['type'], 'value': None}
            b[k]['value'] = self.b[k]['value'] + v['value']
        return cellData(c, b)

    def __sub__(self, other):
        c = self.c - other.c
        b = {}
        for p in other.b.items():
            k, v = p
            b[k] = {'type': v['type'], 'value': None}
            b[k]['value'] = self.b[k]['value'] - v['value']
        return cellData(c, b)

    def __truediv__(self, other):
        c = self.c / other.c
        b = {}
        for p in other.b.items():
            k, v = p
            b[k] = {'type': v['type'], 'value': None}
            b[k]['value'] = self.b[k]['value'] / v['value']
        return cellData(c, b)

    def __mul__(self, other):
        c = self.c * other.c
        b = {}
        for p in other.b.items():
            k, v = p
            b[k] = {'type': v['type'], 'value': None}
            b[k]['value'] = self.b[k]['value'] * v['value']
        return cellData(c, b)

    def inv(self):
        c = 1 / self.c
        b = {}
        for p in self.b.items():
            k, v = p
            b[k] = {'type': v['type'], 'value': None}
            b[k]['value'] = 1. / self.b[k]['value']
        return cellData(c, b)

    def copy(self):
        c =  self.c
        b = {}
        for p in self.b.items():
            k, v = p
            b[k] = {'type': v['type'], 'value': None}
            try:
                b[k]['value'] = self.b[k]['value']
            except KeyError:
                pass
        return cellData(c, b)

class matrixData:
    def __init__(self, lowerV, upperV, diagV, source, internalCoeffs, boundaryCoeffs):
        self.lowerV = lowerV
        self.upperV = upperV
        self.diagV = diagV
        self.source = source
        self.internalCoeffs = internalCoeffs
        self.boundaryCoeffs = boundaryCoeffs


    def __add__(self, other):
        lowerV = self.lowerV + other.lowerV
        upperV = self.upperV + other.upperV
        diagV = self.diagV + other.diagV
        source = self.source + other.source
        
        internalCoeffs = {}
        boundaryCoeffs = {}
        for bc in self.internalCoeffs.items():
            internalCoeffs[bc[0]] = self.internalCoeffs[bc[0]] + other.internalCoeffs[bc[0]]
            boundaryCoeffs[bc[0]] = self.boundaryCoeffs[bc[0]] + other.boundaryCoeffs[bc[0]]
        return matrixData(lowerV, upperV, diagV, source, internalCoeffs, boundaryCoeffs)


    def __sub__(self, other):
        lowerV = self.lowerV - other.lowerV
        upperV = self.upperV - other.upperV
        diagV = self.diagV - other.diagV
        source = self.source - other.source
        
        internalCoeffs = {}
        boundaryCoeffs = {}
        for bc in self.internalCoeffs.items():
            internalCoeffs[bc[0]] = self.internalCoeffs[bc[0]] - other.internalCoeffs[bc[0]]
            boundaryCoeffs[bc[0]] = self.boundaryCoeffs[bc[0]] - other.boundaryCoeffs[bc[0]]
        return matrixData(lowerV, upperV, diagV, source, internalCoeffs, boundaryCoeffs)


    def __neg__(self):
        lowerV = -self.lowerV
        upperV = -self.upperV
        diagV = -self.diagV
        source = -self.source
        internalCoeffs, boundaryCoeffs = {}, {}
        for bc in self.internalCoeffs.items():
            internalCoeffs[bc[0]] = -self.internalCoeffs[bc[0]]
            boundaryCoeffs[bc[0]] = -self.boundaryCoeffs[bc[0]]
        return matrixData(lowerV, upperV, diagV, source, internalCoeffs, boundaryCoeffs)


    def __eq__(self, source):
        self.source += source
        return self

