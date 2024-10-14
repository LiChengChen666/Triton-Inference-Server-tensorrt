import numpy as np
import h5py as h5
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy


fType = 'f'
idType = 'i8'
charType = 'uint8'


def append_dataset(dset, array):
    originalLength = dset.shape[0]
    dset.resize(originalLength + array.shape[0], axis=0)
    dset[originalLength:] = array

def get_cell(path):
    reader = vtk.vtkHDFReader()
    reader.SetFileName(path)
    #reader.SetFileName('/home/data/19010/test/576.hdf')
    reader.SetStep(0)
    reader.Update()
    output = reader.GetOutput()
    points = output.GetPoints()

    vertices = np.array(
        [points.GetPoint(i) for i in range(points.GetNumberOfPoints())]
    )
        
    cells = output.GetCells()  # 或者根据不同单元类型用 GetCellTypes() 过滤
    connectivity = cells.GetConnectivityArray()

    # 获取 Offset 数组
    offset = cells.GetOffsetsArray()

    # 将 Connectivity 和 Offset 转换为 NumPy 数组（可选）
    connectivity_array = np.array([connectivity.GetValue(i) for i in range(connectivity.GetNumberOfValues())])
    offset_array = np.array([offset.GetValue(i) for i in range(offset.GetNumberOfValues())])
    num_cells = output.GetNumberOfCells()
    cell_types = []
    for i in range(num_cells):
        cell_type = output.GetCellType(i)
        cell_types.append(cell_type)

    # 转换为 NumPy 数组（可选）
    cell_types_array = np.array(cell_types)
    
    # 打印单元类型数组
    print("Cell Types:", cell_types_array)

    print("Connectivity Array:", connectivity_array)
    print("Offset Array:", offset_array)
    return connectivity_array,cell_types_array,offset_array,num_cells,vertices
                     
class vtkHDFWriter():
    def __init__(self,fileName,point_data_dict) -> None:
        f = h5.File(fileName,'w')
        self.root = f.create_group('VTKHDF')
        self.steps = self.generate_structure_for_unstructured(self.root)
        self.create_point_data(point_data_dict)
        
    def init_geometry(self,con:np.array,Types:np.array,Offsets:np.array,NumberOfCells:int):
        append_dataset(self.root["Connectivity"], con)
        append_dataset(
            self.root["NumberOfConnectivityIds"], np.array([len(con)])
        )
        append_dataset(self.root["Types"], Types)
        append_dataset(self.root["Offsets"], Offsets)
        append_dataset(self.root["NumberOfCells"], np.array([NumberOfCells]))
    
    def append(self,time_value:float,points:np.array,point_data:dict):

        # append_dataset(self.root["Connectivity"], con)
        # append_dataset(
        #     self.root["NumberOfConnectivityIds"], np.array([len(con)])
        # )
        # append_dataset(self.root["Types"], Types)
        # append_dataset(self.root["Offsets"], Offsets)
        # append_dataset(self.root["NumberOfCells"], np.array([NumberOfCells]))
        
        append_dataset(self.steps["Values"], np.array([time_value]))
        append_dataset(self.steps['PartOffsets'], np.array([0]))
        # append_dataset(self.steps['NumberOfParts'], np.array([0]))
        append_dataset(self.steps['CellOffsets'], np.array([0]))
        append_dataset(self.steps['ConnectivityIdOffsets'], np.array([0]))
        append_dataset(self.root["Points"], points)
        append_dataset(self.root["NumberOfPoints"], np.array([len(points)]))
        append_dataset(
            self.steps["PointOffsets"], np.array([len(points) * self.steps.attrs['NSteps']])
        )
        for key, value in point_data.items():
            append_dataset(self.pointData[key],value.astype('float32'))
            append_dataset(
                self.steps['PointDataOffsets'][key], np.array([len(points) * self.steps.attrs['NSteps']])
            )
        self.steps.attrs['NSteps'] += 1
        
    def append_geometry(self):
        append_dataset(
            self.root['Connectivity'],          
            )
        append_dataset(
            self.root["NumberOfConnectivityIds"], np.array([len(con)])
            )
        append_dataset(
            self.root['NumberOfPoints'], 
                    np.array([part.GetNumberOfPoints()])
            )
        append_dataset(
            self.root['Points'],
                    vtk_to_numpy(part.GetPoints().GetData())
            )
        
        append_dataset(steps["PointOffsets"], 
                    np.array([currentNumberOfPointCoords]))


        # add zeros for connectivity offsetting
        append_dataset(self.steps['CellOffsets'], 
                    np.array([0, 0, 0, 0]).reshape(1, 4))
        append_dataset(self.steps['ConnectivityIdOffsets'], 
                    np.array([0, 0, 0, 0]).reshape(1, 4))


    def append_time(self, time_value: float):
        self.steps.attrs['NSteps'] += 1
        append_dataset(self.steps["Values"], np.array([time_value]))
        append_dataset(self.steps['PartOffsets'], np.array([0]))
        append_dataset(self.steps['NumberOfParts'], np.array([0]))
        append_dataset(self.steps['CellOffsets'], np.array([0]))
        append_dataset(self.steps['ConnectivityIdOffsets'], np.array([0]))

        
    def create_point_data(self, point_data_dict:dict):
        # print(field_data, field_name)
        for key in point_data_dict:
            print(key)
            c = point_data_dict[key]
            if c != 1:
                self.root['PointData'].create_dataset(
                        key, (0, c), maxshape=(None, c), dtype=fType
                    )
            else:
                self.root['PointData'].create_dataset(
                        key, (0,), maxshape=(None,), dtype=fType
                    )
            self.steps['PointDataOffsets'].create_dataset(
                    key, (0,), maxshape=(None,), dtype=idType
                )
    def generate_structure_for_unstructured(self,root):
        root.attrs['Version'] = (2,1)
        ascii_type = 'UnstructuredGrid'.encode('ascii')
        root.attrs.create('Type', ascii_type, dtype=h5.string_dtype('ascii', len(ascii_type)))

        root.create_dataset('Connectivity', (0,), maxshape=(None,), dtype=idType)
        root.create_dataset('NumberOfCells', (0,), maxshape=(None,), dtype=idType)
        root.create_dataset('NumberOfConnectivityIds', (0,), maxshape=(None,), dtype=idType)
        root.create_dataset('NumberOfPoints', (0,), maxshape=(None,), dtype=idType)
        root.create_dataset('Offsets', (0,), maxshape=(None,), dtype=idType)
        root.create_dataset('Points', (0,3), maxshape=(None,3), dtype=fType)
        root.create_dataset('Types', (0,), maxshape=(None,), dtype=charType)
        # root.create_dataset('NumberOfParts', (0,), maxshape=(None,), dtype=idType)
        
        root.create_group('FieldData')
        root.create_group('CellData')
        self.pointData = root.create_group('PointData')

        steps = root.create_group('Steps')
        steps.attrs['NSteps'] = 0

        steps.create_dataset('PartOffsets', (0,), maxshape=(None,), dtype=idType)


        steps.create_dataset('CellOffsets', (0,), maxshape=(None,), dtype=idType)
        steps.create_dataset('ConnectivityIdOffsets', (0,), maxshape=(None,), dtype=idType)

        steps.create_dataset('PointOffsets', (0,), maxshape=(None,), dtype=idType)
        steps.create_dataset('Values', (0,), maxshape=(None,), dtype=fType)

        steps.create_group('CellDataOffsets')
        steps.create_group('FieldDataOffsets')
        steps.create_group('PointDataOffsets')
        return steps
    
if __name__ =='__main__':
    h = vtkHDFWriter('/data/jueyuan/project/meshROM/h5/o1.hdf',{'Effective Stress':1,'Fluid Velocity':3})
    reader = vtk.vtkHDFReader()
    reader.SetFileName('/data/jueyuan/project/meshROM/h5/out1.hdf')
    reader.SetStep(0)
    reader.Update()
    output = reader.GetOutput()
    cells = output.GetCells()  # 或者根据不同单元类型用 GetCellTypes() 过滤
    connectivity = cells.GetConnectivityArray()

    # 获取 Offset 数组
    offset = cells.GetOffsetsArray()

    # 将 Connectivity 和 Offset 转换为 NumPy 数组（可选）
    connectivity_array = np.array([connectivity.GetValue(i) for i in range(connectivity.GetNumberOfValues())])
    offset_array = np.array([offset.GetValue(i) for i in range(offset.GetNumberOfValues())])
    num_cells = output.GetNumberOfCells()
    cell_types = []
    for i in range(num_cells):
        cell_type = output.GetCellType(i)
        cell_types.append(cell_type)

    # 转换为 NumPy 数组（可选）
    cell_types_array = np.array(cell_types)

    # 打印单元类型数组
    print("Cell Types:", cell_types_array)

    print("Connectivity Array:", connectivity_array)
    print("Offset Array:", offset_array)
    h.init_geometry(connectivity_array,cell_types_array,offset_array,num_cells)
    for i in range(2):
        reader.SetStep(i)
        reader.Update()
        output = reader.GetOutput()
        points = vtk_to_numpy(output.GetPoints().GetData())
        time = reader.GetTimeValue()
        Stress = vtk_to_numpy(output.GetPointData().GetArray('Effective Stress'))
        Velocity = vtk_to_numpy(output.GetPointData().GetArray('Fluid Velocity'))
        d = {'Effective Stress':Stress,'Fluid Velocity':Velocity}
        h.append(time,points,d)