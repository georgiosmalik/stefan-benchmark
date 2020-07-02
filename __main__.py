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

import sim.stefan_benchmark

from pathlib import Path

# Toto nastavi cwd na cestu z ktere se pak mohu odkazovat relativne na dalsi slozky:
if os.path.dirname(__file__):
    os.chdir(os.path.dirname(__file__))

if __name__ == "__main__":
    dim=int(sys.argv[1][0])

    # Creates directories for output:
    Path("./out/data/"+str(dim)+"d").mkdir(parents=True,exist_ok=True)
    Path("./out/fig/"+str(dim)+"d").mkdir(parents=True,exist_ok=True)

    # Pokyny k pousteni:
    # pustit prikazem python3 stefan-benchmark 1d z nadrazene slozky
    # kod je pomerne slozity, porad se jedna o pracovni verzi, tedy odpust prosim ten neporadek
    # nize se nastavuji konstanty ktere ovlivnuji vypocet (prvni souvisi s casovym krokem, druhe s sirkou mushy regionu, treti s prostorovou diskretizace, ta treti ted koresponduje s 1d variantou, zbyle jsou univerzalni)
    sim.stefan_benchmark.C_CFL=1.
    sim.stefan_benchmark.em.C_EPS=1.
    sim.stefan_benchmark.prm.meshres=500

    # Run simulation:
    sim.stefan_benchmark.DIM=dim
    sim.stefan_benchmark.splt.DIM=dim
    sim.stefan_benchmark.stefan_benchmark()
