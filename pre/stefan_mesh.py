# ------------------------------
# Stefan benchmark preprocessing
# ------------------------------

import pre.mesh as msh
import sim.params as prm

def stefan_mesh(dim):
    dimswitch = {
        1:msh.mesh1d,
        2:msh.mesh2d,
        3:msh.meshxml
        }
    return dimswitch.get(dim,"Please enter 1d, 2d, or 3d.")
