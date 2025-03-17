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

def main():
    #粒子总数
    num_particles = list(range(100,5100,100))
    #粒子的总面积/盒子面积为packing_density
    packing_density_0=[0.1,0.2,0.3,0.4,0.5]
    #packing_density = packing_density_0 * num_particles/5000
    shape='B'
    gpu=hoomd.device.GPU()
    mc = hoomd.hpmc.integrate.SimplePolygon(default_d=14,default_a=18)
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

    device=gpu
    
    for k in range(10):
        
        for j in packing_density_0:

            for i in num_particles:

                packing_density = j * i /5000

                #创建实例
                system=System(
                    num=i,packing_density=packing_density,
                    packing_density_0=j,
                    particle_area=particle_area,
                    mc=mc,
                    device=device,shape=shape
                )

                system.simulation = hoomd.Simulation(device=device,seed=random.randint(1,10000))
                system.simulation.operations.integrator = mc
                system.simulation.create_state_from_gsd(filename='./gsd_concave/P_concave_{:.2f}_{}_{}.gsd'.format(j,i,k+1))

                logger = hoomd.logging.Logger()
                logger.add(system.mc, quantities=['type_shapes'])
                gsd_writer = hoomd.write.GSD(filename='./gsd_concave/P_concave_{:.2f}_{}_{}.gsd'.format(system.packing_density_0,system.num,k+1),
                                            trigger=hoomd.trigger.Periodic(100),
                                            mode='wb',filter=hoomd.filter.All(),
                                            logger=logger)
                system.simulation.operations.writers.append(gsd_writer)
                start_time=time.time()
                system.simulation.run(1000)
                end_time = time.time()
                print(f"堆叠密度为{j},粒子数目为{i}的系统预热完成，耗时: {end_time - start_time:.2f} 秒")

if __name__ == '__main__':
    main()