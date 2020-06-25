# Tento skript poksytuje data k urceni vyvoje chyby se zjemnujici se siti

import os
import sys
import csv

import sim.stefan_benchmark
import sim.params as prm

if os.path.dirname(__file__):
    os.chdir(os.path.dirname(__file__))

dim=int(sys.argv[1][0])

meshres={
    1:[10,100,1000,10000],
    2:[20,30,40],
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
    
    # Run simulation:
    sim.stefan_benchmark.DIM=dim
    sim.stefan_benchmark.stefan_benchmark()
