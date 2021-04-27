import tkinter as tk
import os
from ofMesh import ofMesh
from ofField import ofField
from ofFVM import FVM, FVC
from ofPISO import PISO

class GUI:
    def __init__(self):
        self.load()
        self.check()

    def load(self):

        window = tk.Tk()
        window.title("icoFoam")
        window.geometry('400x160')

        # Information
        tk.Label(window, text='   Working Directory:', font=('Arial', 12)).place(x=10, y=10)
        tk.Label(window, text='Kinematic Viscosity:', font=('Arial', 12)).place(x=10, y=40)
        tk.Label(window, text='            Stating Time:', font=('Arial', 12)).place(x=10, y = 70)
        tk.Label(window, text='            Ending Time:', font=('Arial', 12)).place(x=10, y = 100)
        tk.Label(window, text='                Time Step:', font=('Arial', 12)).place(x=10, y = 130)

        # Input
        var_caseDir = tk.StringVar()
        var_caseDir.set('..\\ico_2d')
        entry_caseDir = tk.Entry(window, textvariable = var_caseDir, font=('Arial', 12))
        entry_caseDir.place(x=155, y = 10)

        var_nu = tk.StringVar()
        var_nu.set('0.01')
        entry_nu = tk.Entry(window, textvariable = var_nu, font=('Arial', 12))
        entry_nu.place(x=155, y = 40)

        var_st = tk.StringVar()
        var_st.set('0')
        entry_st = tk.Entry(window, textvariable = var_st, font=('Arial', 12))
        entry_st.place(x=155, y = 70)

        var_et = tk.StringVar()
        var_et.set('0.02')
        entry_et = tk.Entry(window, textvariable = var_et, font=('Arial', 12))
        entry_et.place(x=155, y = 100)

        var_del = tk.StringVar()
        var_del.set('0.01')
        entry_del = tk.Entry(window, textvariable = var_del, font=('Arial', 12))
        entry_del.place(x=155, y = 130)

        def set_caseDir():
            if not os.path.isdir(var_caseDir.get()):
                print('This path is invalid')
                self.path = None
                self.startT = None
                self.endT = None
                self.stepT = None
                self.nu = None
            else:
                print('Path: ', var_caseDir.get())
                self.path = var_caseDir.get()
                self.startT = var_st.get()
                self.endT = var_et.get()
                self.stepT = var_del.get()
                self.nu = float(var_nu.get())
                window.destroy()

        start_bn = tk.Button(window, text='Start', font=('Arial', 12), command=set_caseDir)
        start_bn.place(x=350, y = 100)
        window.mainloop()

    def check(self, meshDir = 'constant', meshName ='polyMesh'):
        try:
            ofField(os.path.join(self.path, meshDir), os.path.join(self.path, self.startT))
            return True
        except:
            return False

control = GUI()
# Default variables
meshDir = 'constant'
meshName = 'polyMesh'
limitCoeff = 0.5
nCorrectors = 3
nNonOrthogonalCorrectors = 1
nu = control.nu

# Time Loop
curT = float(control.startT)
while curT <= float(control.endT):
    print('Current Time: {:0.04f}s'.format(curT))
    if curT == float(control.startT): # Load initial values from files
        field = ofField(os.path.join(control.path, meshDir), os.path.join(control.path, control.startT), float(control.stepT))
        fvm = FVM(field, 'UEqn')
        fvc = FVC(field, 'UEqn')
        curT += float(control.stepT)
        continue

    # Solve Navier-Stokes Equation for velocity
    fvm.solve(fvm.dudt() + fvm.div('phi, U') - fvm.laplacian(nu, 'U', limitCoeff) == -fvc.grad('P'))
    # PISO
    PISO(fvm, nCorrectors, nNonOrthogonalCorrectors, limitCoeff)

    if curT == round(float(control.endT), 4):
        field.writeField(curT)

    field.updateField()
    curT += float(control.stepT)
    curT = round(curT, 4)






