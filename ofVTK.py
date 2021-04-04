import vtk
import numpy as np

class Render:
    def __init__(self, filename = 'cavity.vtk'):
        self.mesh = self.readVTK(filename)

    def readVTK(self, filename):
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        ug = reader.GetOutput()
        return ug
    '''
    def assignCellID(self, name, cP = np.arange(self.mesh.GetNumberOfCells())):
        Scalar = vtk.vtkIntArray()
        Scalar.SetName(name)
        Scalar.SetNumberOfTuples(self.mesh.GetNumberOfCells())
        for i in range(self.mesh.GetNumberOfCells()):
            Scalar.SetTuple1(i, cP[i])
        self.mesh.GetCellData().AddArray(Scalar)
    '''
    def writeVTK(self, filename):
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(self.mesh)
        writer.Write()
        
    def assignVector(self, name, cU):
        Vel = vtk.vtkIntArray()
        Vel.SetName(name)
        Vel.SetNumberOfTuples(self.mesh.GetNumberOfCells())
        Vel.SetNumberOfComponents(3)
        for i in range(self.mesh.GetNumberOfCells()):
            Vel.SetTuple3(i, cU[i])
        self.mesh.GetCellData().AddArray(Vel)
        
    def display(self):
        # Grid
        mapper1 = vtk.vtkDataSetMapper()
        mapper1.SetInputData(self.mesh)
        
        actor1 = vtk.vtkActor()
        actor1.SetMapper(mapper1)
        actor1.GetProperty().SetColor(0, 0, 0.1)
        actor1.GetProperty().SetRepresentationToWireframe()
        actor1.GetProperty().SetEdgeColor(0.2, 0, 0)
        actor1.GetProperty().EdgeVisibilityOn()
        
        # Velocity
        arrow = vtk.vtkArrowSource()
        arrow.SetTipResolution(16)
        arrow.SetTipLength(0.3)
        arrow.SetTipRadius(0.1)
        
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(self.mesh)
        glyph.SetSourceConnection(arrow.GetOutputPort())
        glyph.SetVectorModeToUseVector()
        glyph.SetColorModeToColorByVector()
        glyph.OrientOn()
        glyph.Update()
        
        mapper2 = vtk.vtkDataSetMapper()
        mapper2.SetInputConnection(glyph.GetOutputPort())
        mapper2.ScalarVisibilityOn()
        # mapper.SetScalarRange(self.mesh.GetCellData().GetVectors().GetRange(-1))
        
        actor2 = vtk.vtkActor()
        actor2.SetMapper(mapper2)
        
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor1)
        renderer.AddActor(actor2)
        renderer.SetBackground(1, 1, 1)
        
        renderer_window = vtk.vtkRenderWindow()
        renderer_window.SetSize(500, 500)
        renderer_window.SetPosition(200, 200)
        renderer_window.AddRenderer(renderer)
        
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(renderer_window)
        interactor.Initialize()
        renderer_window.Render()
        interactor.Start()
        
    '''
    def display(self):
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.5, 0.5, 0.5)
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        
        arrow = vtk.vtkArrowSource()
        arrow.SetTipResolution(16)
        arrow.SetTipLength(0.3)
        arrow.SetTipRadius(0.1)
        
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(self.mesh)
        glyph.SetSourceConnection(arrow.GetOutputPort())
        glyph.SetVectorModeToUseVector()
        glyph.SetColorModeToColorByVector()
        glyph.OrientOn()
        glyph.Update()
        
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.ScalarVisibilityOn()
        # mapper.SetScalarRange(self.mesh.GetCellData().GetVectors().GetRange(-1))
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        renderer.AddActor(actor)
        renderWindow.Render()
        renderWindowInteractor.Start()
    '''

render = Render()
# render.assignCellID()
# render.writeVTK('cavity_after.vtk')
render.display()
