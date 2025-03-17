import math
import hoomd
import numpy as np
import itertools
import gsd.hoomd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from system_concave import System
import time
import random
from multiprocessing import Pool, cpu_count

def task(n):
    #粒子总数
    num_particles = 5000
    #每次开始插入之前，先对基础的系统预处理pre_random步数
    pre_random = 5000
    #粒子的总面积/盒子面积为packing_density
    packing_density_0= (1+n//900)*0.1
    #packing_density = packing_density_0 * num_particles/5000
    #压缩系数
    condensed_ratio=0.98
    #微小间距
    margin=0.2
    device=hoomd.device.CPU()
    shape='B'
    mc = hoomd.hpmc.integrate.SimplePolygon(default_d=1,default_a=0.5)
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

    #for i in num_particles:

    #for j in packing_density_0:

    packing_density = packing_density_0 * num_particles /5000

    #创建实例
    system=System(
        num=num_particles,packing_density=packing_density,
        packing_density_0=packing_density_0,
        particle_area=particle_area,
        mc=mc,condensed_ratio=condensed_ratio,margin=margin,
        pre_random=pre_random,device=device,shape=shape,sdf_file_num=n%900+1+100
    )
    system.simulation = hoomd.Simulation(device=device,seed=random.randint(1,100))
    system.simulation.operations.integrator = mc
    system.simulation.create_state_from_gsd(filename='gsd_sdf/concave/P_concave_{:.2f}_{}_{}.gsd'.format(packing_density_0,num_particles,n%900+1+100))

    #print(f"\n正在预热系统，进行 {pre_random} 次移动...")
    #start_time=time.time()
    system.save_to_gsd_sdf()
    #system.randomizing_particles()
    #end_time = time.time()
    #print(f"预热完成，耗时: {end_time - start_time:.2f} 秒")

    start_time=time.time()
    for _ in range(10):
        system.simulation.run(100)
    end_time = time.time()
    print(f"堆叠密度为{packing_density_0:.1f},粒子数目为{num_particles}的系统预热完成，耗时: {end_time - start_time:.2f} 秒")

def main():
    if __name__ == '__main__':
        data = list(range(4500)) 
        num_workers = cpu_count()

        with Pool(processes=num_workers) as pool:
            pool.map(task, data)

if __name__ == '__main__':
    main()