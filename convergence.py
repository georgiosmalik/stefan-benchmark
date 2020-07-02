# Tento skript poksytuje data k urceni vyvoje chyby se zjemnujici se siti
# Musim poustet postupne, newtonuv solver nechce konvergovat pote, co spoctu prvni ulohu, kdyz poustim postupne z konzole, tak konverguje naprosto normalne

import os
import subprocess
import sys
import csv

import sim.stefan_benchmark as sbm
import sim.params as prm

from importlib import reload

if os.path.dirname(__file__):
    os.chdir(os.path.dirname(__file__))

dim=int(sys.argv[1][0])
sbm.DIM=dim

meshres={
    1:[100,1000,10000],
    2:[25],
    3:[0.05,0.025,0.01]
    }

# Checking for old data file:
for file in  os.listdir('./out/data/'+str(dim)+'d'):
    if "convergence" in file:
        os.remove('./out/data/'+str(dim)+'d/' + file)

with open('./out/data/'+str(dim)+'d/convergence.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)

    # Zapisujeme typ metody, L2 a L inf normu rel chyby
    filewriter.writerow(['params',
                         'method',
                         'l2norm',
                         'linfnorm',
                         'deltas']
    )

for nx in meshres[dim]:
    prm.meshres=nx

    #if nx==1000:
        #sbm.dolfin.set_log_level(10)
    #sbm=reload(sbm)
    # Run simulation:
    sbm.stefan_benchmark()
    #subprocess.call(['dijitso','clean'])

    
