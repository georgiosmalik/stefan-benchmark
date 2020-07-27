#!/bin/bash
#------------------------------------------------------------
# 3D radially symmetric Stefan problem mesh generation script
#------------------------------------------------------------
# This script generates a mesh in the .xml format from .msh file, dolfin-convert is deprecated in the next versions of FEniCS, use MeshIO instead, run with ./generate_mesh.sh, once the .xml file is generated this does not need to be run again
gmsh ./pre/gmsh_mesh/generate_mesh.gmsh -2
dolfin-convert ./pre/gmsh_mesh/sphere.msh ./pre/gmsh_mesh/sphere.xml
