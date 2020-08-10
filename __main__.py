#-----------------
# stefan-benchmark
#-----------------

# K tvorbe programu jsem se snazil pristupovat filozofii free functions, tedy cely program je cleneny do funkci, ktere maji neco na vstupu a neco vraceji. Tyto funkce by mely odpovidat logickemu cleneni celeho behu programu, navic struktura je viceurovnova, nejhrubsi uroven deli program na preprocessing, processign a postprocessing, toto se zrcadli ve strukture souborove, resp. adresarove. V jednotlivych modulech jsou pak obsazeny free functions ktere napr. obstaravaji generovani site a prislutsnych entit (okraje, normala, integracni miry). Tyto se bud stepi na dalsi funkce, nebo obstaravaji pozadovane vystupy pomoci prikazu dolfinu.
# Tady se spousti benchmark, prikazem python3 stefan_benchmark xd se spusti xd rozmenrny benchmark
# Kod je rozdelen do tri zakladnich struktur:
# Preprocessing: tady se generuje sit, oznacuji se hranice, generuje se normala a integracni miry
# Processing: samotna simulace, obsahuje formulaci slabeho problemu, jeho reseni pro casovou evoluci
# Postprocessing: procedury pro ukladani a vykreslovani vyledku

import os
import sys
import time

import sim.stefan_benchmark

from pathlib import Path

# Toto nastavi cwd na cestu z ktere se pak mohu odkazovat relativne na dalsi slozky:
if os.path.dirname(__file__):
    os.chdir(os.path.dirname(__file__))

if __name__ == "__main__":

    # Set dimension of simulation
    try:
        dim=int(sys.argv[-1][0])
    except ValueError:
        print("Dimension not specified, running "+str(sys.argv[-1])+" 1d.")
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
    sim.stefan_benchmark.R_START = sim.stefan_benchmark.prm.R1 + 0.2
    sim.stefan_benchmark.R_END = sim.stefan_benchmark.prm.R2 - 0.2
    
        
    if "convergence" in sys.argv:
        
        # Run convergence:
        sim.stefan_benchmark.CONVERGENCE=True
        sim.stefan_benchmark.stefan_convergence()
        
    elif "stability" in sys.argv:
        
        # Run stability:
        sim.stefan_benchmark.STABILITY=True
        sim.stefan_benchmark.stefan_stability()
        
    elif "postprocessing-benchmark" in sys.argv:
        
        # Run postprocessing (benchmark data):
        sim.stefan_benchmark.splt.load_data()
        sim.stefan_benchmark.splt.graph_temp()
        sim.stefan_benchmark.splt.graph_front_pos()
        sim.stefan_benchmark.splt.graph_front_vel()

    elif "postprocessing-stability" in sys.argv:

        # Run postprocessing (stability data):
        sim.stefan_benchmark.splt.load_data_stability()
        sim.stefan_benchmark.splt.graph_stability()

    elif "preprocessing" in sys.argv:

        # Build mesh (only 3d):
        sim.stefan_benchmark.smsh.BUILD_MESH_HDF=True
        sim.stefan_benchmark.smsh.stefan_mesh(3)()

    # TEST (tuning 3d benchmark)
    elif "projection" in sys.argv:
        # Run projected simulation:
        sim.stefan_benchmark.PROJECT=True
        sim.stefan_benchmark.GRAPH=True
        sim.stefan_benchmark.SAVE_DAT=False
        sim.stefan_benchmark.stefan_projection()
    #---------------------------

    else:
        # nize se nastavuji konstanty ktere ovlivnuji vypocet (prvni souvisi s casovym krokem, druhe s sirkou mushy regionu, treti s prostorovou diskretizace, ta treti ted koresponduje s 1d variantou, zbyle jsou univerzalni)
        
        # Run simulation:
        sim.stefan_benchmark.BENCHMARK=True
        
        sim.stefan_benchmark.GRAPH=False
        sim.stefan_benchmark.SAVE_DAT=True
        
        solvestart = time.time()
        sim.stefan_benchmark.stefan_benchmark()
        solveend = time.time()
        print("time:", solveend-solvestart)
        
# Pokyny k pousteni:
# pustit prikazem python3 stefan-benchmark 1d z nadrazene slozky
# kod je pomerne slozity, porad se jedna o pracovni verzi, tedy odpust prosim ten neporadek

