import math
import hoomd
import numpy as np
import itertools
import gsd.hoomd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from system import System
import time

def main():
    #粒子总数
    num_particles = 100
    #每次开始插入之前，先对基础的系统预处理pre_random步数
    pre_random = 1000
    #粒子的总面积/盒子面积为packing_density
    packing_density_0=0.1
    #packing_density = packing_density_0 * num_particles/5000
    #压缩系数
    condensed_ratio=0.9
    #微小间距
    margin=0.2
    gpu=hoomd.device.GPU()
    shape='A'
    mc = hoomd.hpmc.integrate.SimplePolygon(default_d=0.5,default_a=0.2)
    mc.shape["A"] = dict(
                vertices = [
                (-2, 0),
                (2, 0),
                (11/8, 5*math.sqrt(63)/8),
                ]
    )
    particle_area=5*math.sqrt(63)/4


    packing_density = packing_density_0 * num_particles /5000

    #创建实例
    system=System(
        num=num_particles,packing_density=packing_density,
        packing_density_0=packing_density_0,
        particle_area=particle_area,
        mc=mc,condensed_ratio=condensed_ratio,margin=margin,
        pre_random=pre_random,device=gpu
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