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
    num_particles = 4000
    #每次开始插入之前，先对基础的系统预处理pre_random步数
    pre_random = 2000
    #粒子的总面积/盒子面积为packing_density
    packing_density_0=0.4
    #packing_density = packing_density_0 * num_particles/5000
    #压缩系数
    condensed_ratio=0.98
    #微小间距
    margin=0.2
    cpu=hoomd.device.CPU()
    shape='B'
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



    packing_density = packing_density_0 * num_particles /5000

    #创建实例
    system=System(
        num=num_particles,packing_density=packing_density,
        packing_density_0=packing_density_0,
        particle_area=particle_area,
        mc=mc,condensed_ratio=condensed_ratio,margin=margin,
        pre_random=pre_random,device=cpu,shape=shape
    )
    system.generate_particle()

    print(f"\n正在预热系统，进行 {pre_random} 次移动...")
    start_time=time.time()
    system.save_to_gsd()
    system.randomizing_particles()
    end_time = time.time()
    print(f"预热完成，耗时: {end_time - start_time:.2f} 秒")

    #for _ in range(5):
    #    system.simulation.run(100)

if __name__ == '__main__':
    main()