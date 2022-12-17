import argparse
import numpy as np
import networkx as nx

parser = argparse.ArgumentParser(description="Process datasets")
parser.add_argument("--datafile", type=str, help="points")
parser.add_argument("--distfile", type=str, help="str")
parser.add_argument("--savefile", type=str)
parser.add_argument("--watts-strogatz", type=bool)
args = parser.parse_args()
if args.watts_strogatz:
    G = nx.Graph()
    file = open('/data/sam/power_grid/opsahl-powergrid/out.opsahl-powergrid', 'r')
    for line in file:
        if line[0] == '%':
            continue
        split_line = line.split()
        v1 = int(split_line[0])
        v2 = int(split_line[1])
        G.add_edge(v1, v2)
    print("Starting to compute distance matrix")
    mat = nx.floyd_warshall_numpy(G)
    print(mat.shape)
    np.save('/data/sam/power_grid/distances.npy', mat)

if args.datafile == '/data/sam/airports/airports.txt':
    G = nx.Graph()
    file = open(args.datafile)
    for line in file:
        split_line = line.split()
        v1 = int(split_line[0])
        v2 = int(split_line[1])
        w = float(split_line[2])
        G.add_edge(v1, v2, weight=w)
        #print(v1, v2)
    
    mat = nx.floyd_warshall_numpy(G)
    print(mat.shape)
    np.save('/data/sam/airports/distances.npy', mat)

if args.datafile == '/data/sam/belfast/belfast/network_combined.csv':
    file = open(args.datafile, 'r')
    G = nx.Graph()
    iter = 0
    lines = file.readlines()
    print("adding nodes to graph")
    
    for line in lines[1:]:
        split_line = line.split(';')
        v1 = int(split_line[0])
        v2 = int(split_line[1])
        w = float(split_line[3])
        G.add_edge(v1, v2, weight=w)
    print("compting distances")
    mat = nx.floyd_warshall_numpy(G)
    print(mat.shape)
    np.save('/data/sam/belfast/distances.npy', mat)

pfile = open(args.datafile, "r")
vectors = []
for x in pfile:
    if x[0] == '#':
        continue
    ptstr = x.split()
    pts = []
    for s in ptstr:
        pts.append(float(s))
    vectors.append(pts)
pfile.close()
dfile = open(args.distfile, "r")
matrix = []
num= 0
row = []
for line in dfile:
    if line[0] == '#':
        continue
    ptstr = line.split()
    
    for s in ptstr:
        row.append(float(s))
        num += 1
    if num == 312:
        matrix.append(row)
        row = []
        num = 0
        
matrix = np.array(matrix)
print(matrix.shape)

np.save('/data/sam/usca312/vectors.npy', vectors)
np.save('/data/sam/usca312/matrix.npy', matrix)
    

