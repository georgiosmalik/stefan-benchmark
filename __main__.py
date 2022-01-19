# ----------------
# Stefan benchmark
# ----------------

import os
import sys
import time

import sim.stefan_benchmark

from pathlib import Path

# Set cwd for relative path links:
if os.path.dirname(__file__):
    os.chdir(os.path.dirname(__file__))

if __name__ == "__main__":

    # Set dimension of simulation
    try:
        dim=int(sys.argv[-1][0])
    except ValueError:
        print("Dimension set to d=1, running "+str(sys.argv[-1])+" 1d.")
        dim=1

    # Creates directories for output:
    Path("./out/data/"+str(dim)+"d").mkdir(parents=True,exist_ok=True)
    Path("./out/fig/"+str(dim)+"d").mkdir(parents=True,exist_ok=True)

    sim.stefan_benchmark.DIM=dim
    sim.stefan_benchmark.splt.DIM=dim

    # Set heat source intensity
    sim.stefan_benchmark.prm.q_0=0.2*10**(3*(3-dim))

    # Set spatial span of simulation (r2 = 1.0)
    sim.stefan_benchmark.prm.R1=0.1*(dim-1)

    # Set timespan of simulation
    # sim.stefan_benchmark.R_START = sim.stefan_benchmark.prm.R1 + 0.2
    # sim.stefan_benchmark.R_END = sim.stefan_benchmark.prm.R2 - 0.2
    
        
    if "convergence" in sys.argv:
        
        # Run convergence:
        sim.stefan_benchmark.CONVERGENCE=True
        sim.stefan_benchmark.stefan_convergence()
        
    elif "stability-1p" in sys.argv:
        
        # Run one-parametric stability benchmark:
        sim.stefan_benchmark.STABILITY=True
        sim.stefan_benchmark.stability1p()

    elif "stability-2p" in sys.argv:
        
        # Run two-parametric stability benchmark:
        sim.stefan_benchmark.STABILITY=True
        sim.stefan_benchmark.stability2p()
        
    elif "postprocessing-benchmark" in sys.argv:
        
        # Run postprocessing (benchmark data):
        sim.stefan_benchmark.splt.load_data()
        sim.stefan_benchmark.splt.graph_temp()
        sim.stefan_benchmark.splt.graph_front_pos()
        sim.stefan_benchmark.splt.graph_front_vel()

    elif "postprocessing-stability-1p" in sys.argv:

        # Run postprocessing (stability data):
        sim.stefan_benchmark.splt.load_data_stability()
        sim.stefan_benchmark.splt.graph_stability1p()

    elif "postprocessing-stability-2p" in sys.argv:

        # Run postprocessing (stability data):
        sim.stefan_benchmark.splt.load_data_stability()
        sim.stefan_benchmark.splt.graph_stability2p()

    elif "preprocessing" in sys.argv:

        # Build mesh with specified parameter, if given
        try:
            nx=float(sys.argv[-1])
            sim.stefan_benchmark.prm.meshres[3]=nx
        except ValueError:
            pass

        # Build mesh (only 3d):
        sim.stefan_benchmark.smsh.BUILD_MESH_HDF=True
        sim.stefan_benchmark.smsh.stefan_mesh(3)()

    else:
        
        # Run benchmark simulation:
        sim.stefan_benchmark.SAVE_DAT = True
        
        solvestart = time.time()
        sim.stefan_benchmark.stefan_benchmark()
        solveend = time.time()
        print("time:", solveend-solvestart)
