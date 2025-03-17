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
    num_particles = list(range(100,5100,100))
    #每次开始插入之前，先对基础的系统预处理pre_random步数
    pre_random = 500
    #粒子的总面积/盒子面积为packing_density
    packing_density_0=[0.1,0.2,0.3,0.4,0.5]
    #packing_density = packing_density_0 * num_particles/5000
    #压缩系数
    condensed_ratio=0.98
    #微小间距
    margin=0.2
    device=hoomd.device.GPU()
    shape='B'
    mc = hoomd.hpmc.integrate.SimplePolygon(default_d=10,default_a=12)
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

    good_d=[[0]*int(5000/100)]*5
    good_a=[[0]*int(5000/100)]*5

    for k in range(5,10,1):

        for j in packing_density_0:

            for i in num_particles:

                packing_density = j * i /5000

                #创建实例
                system=System(
                    num=i,packing_density=packing_density,
                    packing_density_0=j,
                    particle_area=particle_area,
                    mc=mc,condensed_ratio=condensed_ratio,margin=margin,
                    pre_random=pre_random,device=device,shape=shape
                )
                start_time=time.time()
                system.generate_system()

                print(f"\n正在预热系统，进行 {pre_random} 次移动...")
                logger = hoomd.logging.Logger()
                logger.add(system.mc, quantities=['type_shapes'])
                gsd_writer = hoomd.write.GSD(filename='./gsd_concave/P_concave_{:.2f}_{}_{}.gsd'.format(system.packing_density_0,system.num,k+1),
                                            trigger=hoomd.trigger.Periodic(100),
                                            mode='wb',filter=hoomd.filter.All(),
                                            logger=logger)
                system.simulation.operations.writers.append(gsd_writer)
                system.simulation.run(pre_random)
                end_time = time.time()
                print(f"预热完成，耗时: {end_time - start_time:.2f} 秒")

                good_d[int(j*10)-1][int(i/100)-1]+=system.mc.d['B']
                good_a[int(j*10)-1][int(i/100)-1]+=system.mc.a['B']

                #for _ in range(5):
                #    system.simulation.run(100)
    for i in range(5):
        for j in range(50):
            good_d[i][j]/=5
            good_a[i][j]/=5

    print(good_d)
    print(good_a)

if __name__ == '__main__':
    main()