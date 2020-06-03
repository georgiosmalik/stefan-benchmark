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

# Toto nastavi cwd na cestu z ktere se pak mohu odkazovat relativne na dalsi slozky:
os.chdir(os.path.dirname(__file__))

if __name__ == "__main__":
    sim.stefan_benchmark.DIM=int(sys.argv[1][0])
    sim.stefan_benchmark.stefan_benchmark()
