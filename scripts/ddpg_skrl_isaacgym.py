from pxr import Usd, UsdGeom

print("USD API is working!")


stage = Usd.Stage.Open("/home/nur/Downloads/dex_cube_instanceable.usd")
cube = stage.GetDefaultPrim()

usd_cube = UsdGeom.Cube(cube)
print("Cube Size:", usd_cube.GetSizeAttr().Get())
