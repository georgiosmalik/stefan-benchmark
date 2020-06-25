# ------------------------------
# Stefan benchmark preprocessing
# ------------------------------

# Poznamky a opravy:
# DONE 1. Zautomatizovat generovani site pro 3d?

import subprocess
import os

import pre.mesh as msh
import sim.params as prm

def stefan_mesh(dim):
    def stefan_mesh_1d():
        prm.q_0=2e5
        
        prm.R1=0
        #prm.meshres=10000
        
        return msh.mesh1d(prm.R1,prm.R2,prm.meshres)
    
    def stefan_mesh_2d():
        
        prm.q_0=2000
        
        prm.R1=0.1
        prm.meshres=20
        
        return msh.mesh2d(prm.R1,prm.R2,prm.meshres)
    
    def stefan_mesh_3d():

        prm.q_0=5
        
        # The mesh was generated as a spherical shell between R1=0.2 and R2=1
        prm.R1=0.2

        
        # Test generating with gmsh script:
        
        subprocess.call(['sed','-i',"/Characteristic Length {6, 5, 4, 2, 3, 1} = .*;/c\Characteristic Length {6, 5, 4, 2, 3, 1} = "+str(prm.nx)+";",'./pre/gmsh_mesh/sphere.geo'])
        
        subprocess.call('./pre/gmsh_mesh/generate_mesh.sh')
        msh.meshxml("./pre/gmsh_mesh/sphere")
        #=================================

        return msh.meshhdf("./pre/gmsh_mesh/sphere")
    
    dimswitch = {
        1:stefan_mesh_1d,
        2:stefan_mesh_2d,
        3:stefan_mesh_3d
        }
    
    return dimswitch.get(dim,"Please enter 1d, 2d, or 3d.")
