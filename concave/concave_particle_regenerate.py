import math
import hoomd
import numpy as np
import itertools
import gsd.hoomd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from system_concave import System
import time

def main():
    #粒子总数
    num_particles = [250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,
                     3000,3250,3500,3750,4000,4250,4500,4750,5000]
    #每次开始插入之前，先对基础的系统预处理pre_random步数
    pre_random = 2000
    #粒子的总面积/盒子面积为packing_density
    packing_density_0=[0.1,0.2,0.3,0.4,0.5]
    #packing_density = packing_density_0 * num_particles/5000
    #压缩系数
    condensed_ratio=0.98
    #微小间距
    margin=0.2
    shape='B'
    cpu=hoomd.device.CPU()
    mc = hoomd.hpmc.integrate.SimplePolygon(default_d=0.5,default_a=0.2)
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

    device=cpu
    
    for i in num_particles:

        for j in packing_density_0:

            packing_density = j * i /5000

            #创建实例
            system=System(
                num=i,packing_density=packing_density,
                packing_density_0=j,
                particle_area=particle_area,
                mc=mc,condensed_ratio=condensed_ratio,margin=margin,
                pre_random=pre_random,
                device=device,shape=shape
            )

            system.simulation = hoomd.Simulation(device=device,seed=1)
            system.simulation.operations.integrator = mc
            system.simulation.create_state_from_gsd(filename='./gsd_concave/P_concave_{:.2f}_{}.gsd'.format(j,i))

            #print(f"\n正在预热系统，进行 {pre_random} 次移动...")
            #start_time=time.time()
            system.save_to_gsd()
            #system.randomizing_particles()
            #end_time = time.time()
            #print(f"预热完成，耗时: {end_time - start_time:.2f} 秒")

            #print(f"\n正在重新预热系统，进行 {500} 次移动...")
            start_time=time.time()
            for _ in range(10):
                system.simulation.run(50)
            end_time = time.time()
            print(f"堆叠密度为{j},粒子数目为{i}的系统预热完成，耗时: {end_time - start_time:.2f} 秒")

if __name__ == '__main__':
    main()