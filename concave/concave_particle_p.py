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
                     3000,3250,3500,3750,4000,4250,4500,4750]
    #每次开始插入之前，先对基础的系统预处理pre_random步数
    pre_random = 2000
    #粒子的总面积/盒子面积为packing_density
    #packing_density_0=0.5
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

    
    for i in packing_density_0:
        result=[]
        for j in num_particles:
            packing_density = i * j/5000
            #创建实例
            system=System(
                num=j,packing_density=packing_density,
                packing_density_0=i,
                particle_area=particle_area,
                mc=mc,condensed_ratio=condensed_ratio,margin=margin,
                pre_random=pre_random,
                device=device,shape=shape
            )

            system.simulation = hoomd.Simulation(device=device,seed=1)
            system.simulation.operations.integrator = mc
            system.simulation.create_state_from_gsd(filename='./gsd_concave/P_concave_{:.2f}_{}.gsd'.format(i,j))

            # 运行循环模拟和插入测试
            iterations = 1000              # 循环次数
            moves_per_cycle = 5            # 每个循环中移动的成功步数
            insertions_per_cycle = 100000    # 每个循环中插入的粒子尝试次数

            total_success = 0
            total_attempts = 0


            print("\n开始进行循环模拟和插入测试...")
            simulation_start_time = time.time()
            for cycle in range(1, iterations + 1):
                # 进行移动
                system.simulation.run(moves_per_cycle)
                
                # 进行插入尝试  
                success = system.random_insert(
                    insert_times=insertions_per_cycle
                )
                total_success += success
                total_attempts += insertions_per_cycle

                # 可选：打印每个循环的结果
                if cycle % (iterations // 10 ) == 0:
                    simulation_interval_time = time.time()
                    print(f"循环 {cycle}/{iterations}: 成功插入 {success}/{insertions_per_cycle} 个粒子;耗时: {simulation_interval_time - simulation_start_time:.2f} 秒")
            
            simulation_end_time = time.time()
            print(f"\n堆叠密度为{i}，粒子数为{j}的循环模拟和插入测试完成，耗时: {simulation_end_time - simulation_start_time:.2f} 秒")
            # 计算最终的成功插入概率
            final_probability = total_success / total_attempts if total_attempts > 0 else 0.0
            print(f"最终插入成功概率: {final_probability * 100:.5f}% ({total_success}/{total_attempts}) ;ln(Pi)={math.log(final_probability)}")

            result.append(math.log(final_probability))

            with open('./result/concave_result.txt','a') as file:
                file.write(f"\n堆叠密度为{i}，粒子数为{j}的循环模拟和插入测试完成，耗时: {simulation_end_time - simulation_start_time:.2f} 秒\n")
                file.write(f"最终插入成功概率: {final_probability * 100:.5f}% ({total_success}/{total_attempts}) ;ln(Pi)={math.log(final_probability)}\n")
                file.write("\n")

        with open('./result/concave_result.txt','a') as file:
            file.write(result)

if __name__ == '__main__':
    main()
