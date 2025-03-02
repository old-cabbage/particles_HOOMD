import math
import hoomd
import numpy as np
import itertools
import gsd.hoomd
import matplotlib.pyplot as plt
from system import System
import time
from multiprocessing import Pool, cpu_count
import random

def task(n):
        #粒子总数
        N=5000
        num_particles = 5000
        #每次开始插入之前，先对基础的系统预处理pre_random步数
        pre_random = 5000
        #粒子的总面积/盒子面积为packing_density
        packing_density_0=0.3
        packing_density = packing_density_0 * num_particles/N
        #压缩系数
        condensed_ratio=0.98
        #微小间距
        margin=0.2
        
        divice=hoomd.device.CPU()
        mc = hoomd.hpmc.integrate.ConvexPolygon(default_d=3,default_a=0.5)
        mc.shape["A"] = dict(
                        vertices = [
                        (-2, 0),
                        (2, 0),
                        (11/8, 5*math.sqrt(63)/8),
                        ]
            )
        particle_area=5*math.sqrt(63)/4

        system=System(
                num=num_particles,packing_density=packing_density,
                packing_density_0=packing_density_0,
                particle_area=particle_area,
                mc=mc,condensed_ratio=condensed_ratio,margin=margin,
                pre_random=pre_random,device=divice
            )

        system.simulation = hoomd.Simulation(device=divice,seed=random.randint(1,10+n))
        system.simulation.operations.integrator = mc
        system.simulation.create_state_from_gsd(filename='./gsd/P_{:.2f}_{}.gsd'.format(packing_density_0,num_particles))

        sdf_mc=1000
        sdf_xmax=0.01
        sdf_dx=1e-4
        sdf_each_run=5

        total_sdf_xcompression,total_sdf_sdfcompression=system.calculate_sdf(sdf_mc,sdf_xmax,sdf_dx,sdf_each_run)

        return total_sdf_xcompression,total_sdf_sdfcompression

def main():

    packing_density_0=0.3
    
    if __name__ == '__main__':
        data = list(range(100)) 
        num_workers = cpu_count()

        with Pool(processes=num_workers) as pool:
            results = pool.map(task, data)
    


    total_sdf_xcompression=np.zeros(100)
    total_sdf_sdfcompression=np.zeros(100)
    
    for i in range(len(results)):
        interval_sdf_xcompression,interval_sdf_sdfcompression=results[i]
        total_sdf_xcompression += interval_sdf_xcompression
        total_sdf_sdfcompression += interval_sdf_sdfcompression

    total_sdf_xcompression/=len(results)
    total_sdf_sdfcompression/=len(results)

    print(total_sdf_xcompression,total_sdf_sdfcompression)

    plt.plot(total_sdf_xcompression,total_sdf_sdfcompression,label=r"$\phi={}$".format(packing_density_0),
                linestyle='-',linewidth=0.5,color='purple'
                )

    plt.plot(total_sdf_xcompression,total_sdf_sdfcompression,marker='o',
            color='purple',
            markersize=10,            # 标记大小
            markerfacecolor='none',   # 标记内部为空心
            markeredgecolor='purple',  # 标记边框颜色为黑色
            markeredgewidth=1,      # 标记边框宽度
            alpha=0.2
            )

    # 使用 scatter 绘制半透明的标记点
    #plt.scatter(total_sdf_xcompression,total_sdf_sdfcompression, color='purple',
    #             s=100, alpha=0.5,markerfacecolor='none',markereredgewidth=1)

    plt.legend()
    # 添加标题和标签
    plt.title('Scale Distribution Function')
    plt.xlabel('distance r')
    plt.ylabel('SDF(r)')
    plt.show()

if __name__ == '__main__':
    main()