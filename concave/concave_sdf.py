import math
import hoomd
import numpy as np
import itertools
import gsd.hoomd
import matplotlib.pyplot as plt
from system_concave import System
import time
from multiprocessing import Pool, cpu_count
import random

#mpi
cyc_num=100
index=3
total_upload=10

packing_density_0=0.5
num_particles = 5000

sdf_mc=50
sdf_xmax=0.01
sdf_dx=1e-4
sdf_each_run=5

#粒子总数
N=5000
#每次开始插入之前，先对基础的系统预处理pre_random步数
pre_random = 5000
packing_density = packing_density_0 * num_particles/N
shape='B'
device=hoomd.device.CPU()
mc = hoomd.hpmc.integrate.SimplePolygon(default_d=2,default_a=0.5)
mc.shape[shape] = dict(
            vertices = [
            (-1, 0),
            (1, 0),
            (1,2),
            (0,1),
            (-2,2)
            ]
)
particle_area=7/2

def task(n):

    system=System(
            num=num_particles,packing_density=packing_density,
            packing_density_0=packing_density_0,
            particle_area=particle_area,
            mc=mc,pre_random=pre_random,device=device,shape=shape
        )

    system.simulation = hoomd.Simulation(device=device,seed=random.randint(1,10000))
    system.simulation.operations.integrator = mc
    system.simulation.create_state_from_gsd(filename='gsd_sdf/concave/P_concave_{:.2f}_{}_{}.gsd'.format(packing_density_0,num_particles,n%100+1))

    print(f"堆叠密度为{packing_density_0},第{n+1}个{sdf_mc}次sdf_mc循环")

    total_sdf_xcompression,total_sdf_sdfcompression,total_sdf_sdfexpansion=system.calculate_sdf(sdf_mc,sdf_xmax,sdf_dx,sdf_each_run)

    return total_sdf_xcompression,total_sdf_sdfcompression,total_sdf_sdfexpansion

def main():

    #packing_density_0=0.3
    
    if __name__ == '__main__':
        data = list(range(cyc_num*(index-1),cyc_num*index,1))
        num_workers = cpu_count()-2

        with Pool(processes=num_workers) as pool:
            results = pool.map(task, data)
    
    total_sdf_xcompression=np.zeros(int(sdf_xmax/sdf_dx))
    total_sdf_sdfcompression=np.zeros(int(sdf_xmax/sdf_dx))
    total_sdf_sdfexpansion=np.zeros(int(sdf_xmax/sdf_dx))
    
    for i in range(len(results)):
        interval_sdf_xcompression,interval_sdf_sdfcompression,interval_sdf_sdfexpansion=results[i]
        total_sdf_xcompression += interval_sdf_xcompression
        total_sdf_sdfcompression += interval_sdf_sdfcompression
        total_sdf_sdfexpansion += interval_sdf_sdfexpansion

    total_sdf_xcompression/=len(results)
    total_sdf_sdfcompression/=len(results)
    total_sdf_sdfexpansion/=len(results)

    print(total_sdf_xcompression.tolist()) 
    print(total_sdf_sdfcompression.tolist())
    print(total_sdf_sdfexpansion.tolist())

    with open("result/sdf/concave_sdf_{:.2f}_{}.txt".format(packing_density_0,num_particles),'a') as file:
        file.write(f"序列{index}/{total_upload},堆叠密度为{packing_density_0},{cyc_num}个{sdf_mc}次sdf_mc循环,调用了{num_workers}个核心\n{total_sdf_xcompression.tolist()}\n{total_sdf_sdfcompression.tolist()}\n{total_sdf_sdfexpansion.tolist()}\n\n")

if __name__ == '__main__':
    main()