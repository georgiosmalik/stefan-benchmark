# ------------------------------
# Stefan benchmark preprocessing
# ------------------------------

# Poznamky a opravy:
# 1. Zautomatizovat generovani site pro 3d?


import pre.mesh as msh
import sim.params as prm

def stefan_mesh(dim):
    def stefan_mesh_1d():
        prm.R1=0
        prm.meshres=10
        return msh.mesh1d(prm.R1,prm.R2,prm.meshres)
    def stefan_mesh_2d():
        prm.R1=0.1
        prm.mshres=20
        return msh.mesh2d(prm.R1,prm.R2,prm.meshres)
    def stefan_mesh_3d():
        # The mesh was generated as a spherical shell between R1=0.2 and R2=1
        prm.R1=0.2
        return msh.meshhdf("./pre/gmsh_mesh/sphere")
    dimswitch = {
        1:stefan_mesh_1d,
        2:stefan_mesh_2d,
        3:stefan_mesh_3d
        }
    return dimswitch.get(dim,"Please enter 1d, 2d, or 3d.")
