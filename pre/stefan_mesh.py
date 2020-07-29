# ------------------------------
# Stefan benchmark preprocessing
# ------------------------------

# Poznamky a opravy:
# DONE 1. Zautomatizovat generovani site pro 3d?

import subprocess
import os

import pre.mesh as msh
import sim.params as prm

BUILD_MESH_HDF=False

def stefan_mesh(dim):
    
    def stefan_mesh_1d():
        
        return msh.mesh1d(prm.R1,prm.R2,prm.meshres[1])
    
    def stefan_mesh_2d():
        
        return msh.mesh2d(prm.R1,prm.R2,prm.meshres[2])
    
    def stefan_mesh_3d():
        
        # The mesh was generated as a spherical shell between R1=0.2 and R2=1

        if BUILD_MESH_HDF:
        
            subprocess.call(['sed','-i',"/Characteristic Length {6, 5, 4, 2, 3, 1} = .*;/c\Characteristic Length {6, 5, 4, 2, 3, 1} = "+str(prm.meshres[3])+";",'./pre/gmsh_mesh/sphere.geo'])
        
            subprocess.call('./pre/gmsh_mesh/generate_mesh.sh')
            
            return msh.meshxml("./pre/gmsh_mesh/sphere")

        return msh.meshhdf("./pre/gmsh_mesh/sphere")
    
    dimswitch = {
        1:stefan_mesh_1d,
        2:stefan_mesh_2d,
        3:stefan_mesh_3d
        }
    
    return dimswitch.get(dim,"Please enter 1d, 2d, or 3d.")
