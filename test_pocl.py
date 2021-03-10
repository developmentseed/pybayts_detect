import pyopencl as cl

plats = cl.get_platforms()
print(plats)
print([plat.version for plat in plats])
