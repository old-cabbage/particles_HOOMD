import math
import hoomd
import numpy as np
import itertools
import time
import gsd.hoomd
import matplotlib.pyplot as plt
import random
import time

#设置使用的设备
cpu = hoomd.device.CPU()

class System:
    def __init__(self,num,packing_density,packing_density_0,particle_area,mc,condensed_ratio=0.98,margin=0.2,pre_random=10,
                 device=cpu,shape='A',sdf_file_num=1):
        self.num=num
        self.packing_density=packing_density
        self.packing_density_0=packing_density_0
        self.particle_area=particle_area
        self.volume = self.num * self.particle_area / self.packing_density
        self.pre_random=pre_random
        self.success_probability = []
        self.device=device
        self.mc=mc
        self.condensed_ratio=condensed_ratio
        self.margin=margin
        self.shape=shape
        self.sdf_file_num=sdf_file_num
        self.mixture=False

    def generate_particle(self):
        if self.shape == 'A':
            #生成一个有N_particles的有序粒子群
            #K = math.ceil(self.num ** (1 / 2))
            L = width = height = math.sqrt(self.num * self.particle_area / self.packing_density)

            xa,ya=self.mc.shape["A"]["vertices"][2]
            xb,yb=self.mc.shape["A"]["vertices"][0]
            xc,yc=self.mc.shape["A"]["vertices"][1]

            side_length_a= xc-xb
            height1 = ya
            x_a = 27/8
            y_a=5/8

            positions=[]
            orientation=[]

            unit_area = width * height / self.num

            # 计算网格间距，确保三角形之间不重叠
            # 使用更紧凑的间距以提高密度
            #row_spacing = self.condensed_ratio * unit_area/(side_length_a + self.margin)    #height1 * 0.75 + margin   紧凑行间距
            col_spacing = side_length_a + self.margin    # 列间距保持不变
            row_spacing = self.condensed_ratio * unit_area/(col_spacing)

            # 计算行列数
            cols = int(width // col_spacing)
            rows = int(height // row_spacing)

                # 计算实际总可放置粒子数量
            total_possible = cols * rows
            if total_possible < self.num:
                print(f"警告：在给定的空间内以要求的间距无法放置所有三角形粒子。")
                print(f"尝试减少粒子数量或增大空间面积，或降低packing_density。")
                self.num = total_possible
                print(f"只能放下 {self.num} 个粒子。")

            count = 0
            for i in range(rows):
                for j in range(cols):
                    if count >= self.num:
                        break
                    # 计算x位置，偶数行不偏移，奇数行偏移半个列间距
                    if i % 2 == 0:
                        x = j * col_spacing - width/2 
                        y = i * row_spacing - height/2
                    else:
                        x = (j + 1 ) * col_spacing - y_a - width/2 
                        y = (i + 0.5) * row_spacing - height/2
                    theta = 0.0 if i % 2 == 0 else math.pi  # 偶数行朝向0，奇数行朝向180度
                    # 确保三角形不会出边界
                    #if (x>=-width/2-xb) and (y>=-height/2) and (x <= width/2 - xc) and (y <= height/2 - ya):
                    if (x>=-width/2) and (y>=-height) and (x <= width/2 ) and (y <= height/2 ):
                        positions.append([x,y,0])
                        orientation.append([np.cos(theta/2), 0, 0, np.sin(theta/2)])
                        count += 1
            print(f"成功初始化了 {len(positions)} 个有序排列的三角形粒子（交错网格）,盒子长{width:.2f},盒子高{height:.2f},行间距={row_spacing:.2f}, 列间距={col_spacing:.2f}。")
       
        elif self.shape=='B':
            num_endpoint=len(self.mc.shape[self.shape]["vertices"])
            endpoints_x=[]
            endpoints_y=[]
            for i in range(num_endpoint):
                x,y=self.mc.shape[self.shape]["vertices"][i]
                endpoints_x.append(x)
                endpoints_y.append(y)
            particle_width=(max(endpoints_x)-min(endpoints_x))
            particle_height=(max(endpoints_y)-min(endpoints_y))
            part_area=self.particle_area*self.condensed_ratio/self.packing_density
            expand_ratio=math.sqrt(part_area/(particle_width*particle_height))
            col_spacing=particle_width*expand_ratio
            row_spacing=particle_height*expand_ratio
            L = width = height = math.sqrt(self.num * self.particle_area / self.packing_density)
            # 计算行列数
            cols = round(width / col_spacing)
            rows = round(height / row_spacing)

            positions=[]
            orientation=[]

            count=0
            for i in range(rows):
                for j in range(cols):
                    if count >= self.num:
                        break
                    x = j * col_spacing - width/2 
                    y = i * row_spacing - height/2
                    positions.append([x,y,0])
                    orientation.append([np.cos(0.0/2), 0, 0, np.sin(0.0/2)])
                    count += 1
            print(f"成功初始化了堆叠密度为{self.packing_density_0} ,数目为 {len(positions)} 有序排列的粒子,盒子长{width:.2f},盒子高{height:.2f},行间距={row_spacing:.2f}, 列间距={col_spacing:.2f}")

        # 取前 N_particles 个位置
        positions = np.array(positions)  # 形状为 (N_particles, 2)
        orientation = np.array(orientation)

        #创建一个快照
        snapshot=hoomd.Snapshot()
        snapshot.particles.N = self.num
        snapshot.particles.position[:] = positions
        snapshot.particles.orientation[:] = orientation
        snapshot.particles.typeid[:] = [0] * self.num
        snapshot.particles.types=[self.shape]
        snapshot.configuration.box=[L, L, 0, 0, 0, 0]
        self.snapshot=snapshot
        self.simulation = hoomd.Simulation(device=self.device,seed=random.randint(1,100))
        self.simulation.operations.integrator = self.mc
        self.simulation.create_state_from_snapshot(self.snapshot)
        
    def generate_system(self):
        if self.mixture == True:

            num_endpoint=len(self.mc.shape['A']["vertices"])
            endpoints_x_A=[]
            endpoints_y_A=[]
            for i in range(num_endpoint):
                x,y=self.mc.shape['A']["vertices"][i]
                endpoints_x_A.append(x)
                endpoints_y_A.append(y)
            col_spacing_A=(max(endpoints_x_A)-min(endpoints_x_A))
            row_spacing_A=(max(endpoints_y_A)-min(endpoints_y_A))

            num_endpoint=len(self.mc.shape['B']["vertices"])
            endpoints_x_B=[]
            endpoints_y_B=[]
            for i in range(num_endpoint):
                x,y=self.mc.shape['B']["vertices"][i]
                endpoints_x_B.append(x)
                endpoints_y_B.append(y)
            col_spacing_B=(max(endpoints_x_B)-min(endpoints_x_B))
            row_spacing_B=(max(endpoints_y_B)-min(endpoints_y_B))

            col_spacing=row_spacing=max(col_spacing_A,col_spacing_B,row_spacing_A,row_spacing_B)*1.2
            cols=rows=math.ceil(math.sqrt(self.num))
            width=col_spacing*(cols+1)
            height=row_spacing*(rows+1)

            positions=[]
            orientation=[]
            typeid=[0]*int(self.num*(1-self.concave_mixture_ratio))+[1]*int(self.num*self.concave_mixture_ratio)
            random.shuffle(typeid)

            count=0
            for i in range(rows):
                for j in range(cols):
                    if count >= self.num:
                        break
                    x = j * col_spacing - width/2 
                    y = i * row_spacing - height/2
                    positions.append([x,y,0])
                    orientation.append([np.cos(0.0/2), 0, 0, np.sin(0.0/2)])
                    count += 1
            print(f"成功初始化了 {len(positions)} 个有序排列粒子,盒子长{width:.2f},盒子高{height:.2f},行间距={row_spacing:.2f}, 列间距={col_spacing:.2f};现在开始压缩体系")
            
            #创建一个快照
            snapshot=hoomd.Snapshot()
            snapshot.particles.N = self.num
            snapshot.particles.position[:] = positions
            snapshot.particles.orientation[:] = orientation
            snapshot.particles.typeid[:] = typeid
            snapshot.particles.types = ['A',"B"]
            snapshot.configuration.box=[width, height, 0, 0, 0, 0]
            self.snapshot=snapshot
            self.simulation = hoomd.Simulation(device=self.device,seed=random.randint(1,10000))
            self.simulation.operations.integrator = self.mc
            self.simulation.create_state_from_snapshot(self.snapshot)
        else:
            num_endpoint=len(self.mc.shape[self.shape]["vertices"])
            endpoints_x=[]
            endpoints_y=[]
            for i in range(num_endpoint):
                x,y=self.mc.shape[self.shape]["vertices"][i]
                endpoints_x.append(x)
                endpoints_y.append(y)
            col_spacing=(max(endpoints_x)-min(endpoints_x))
            row_spacing=(max(endpoints_y)-min(endpoints_y))
            col_spacing=row_spacing=max(col_spacing,row_spacing)*1.2
            cols=rows=math.ceil(math.sqrt(self.num))
            width=col_spacing*(cols+1)
            height=row_spacing*(rows+1)

            positions=[]
            orientation=[]

            count=0
            for i in range(rows):
                for j in range(cols):
                    if count >= self.num:
                        break
                    x = j * col_spacing - width/2 
                    y = i * row_spacing - height/2
                    positions.append([x,y,0])
                    orientation.append([np.cos(0.0/2), 0, 0, np.sin(0.0/2)])
                    count += 1
            print(f"成功初始化了 {len(positions)} 个有序排列粒子,盒子长{width:.2f},盒子高{height:.2f},行间距={row_spacing:.2f}, 列间距={col_spacing:.2f};现在开始压缩体系")
            
            #创建一个快照
            snapshot=hoomd.Snapshot()
            snapshot.particles.N = self.num
            snapshot.particles.position[:] = positions
            snapshot.particles.orientation[:] = orientation
            snapshot.particles.typeid[:] = [0]*self.num
            snapshot.particles.types = [self.shape]
            snapshot.configuration.box=[width, height, 0, 0, 0, 0]
            self.snapshot=snapshot
            self.simulation = hoomd.Simulation(device=self.device,seed=random.randint(1,10000))
            self.simulation.operations.integrator = self.mc
            self.simulation.create_state_from_snapshot(self.snapshot)

        #压缩体系
        initial_box = self.simulation.state.box
        final_box = hoomd.Box.from_box(initial_box)
        final_box.volume = self.simulation.state.N_particles * self.particle_area / self.packing_density
        compress = hoomd.hpmc.update.QuickCompress(
            trigger=hoomd.trigger.Periodic(10), target_box=final_box
        )
        self.simulation.operations.updaters.append(compress)
        periodic = hoomd.trigger.Periodic(10)
        tune = hoomd.hpmc.tune.MoveSize.scale_solver(
            moves=["a", "d"],
            target=0.2,
            trigger=periodic,
            max_translation_move=5,
            max_rotation_move=5,
        )
        self.simulation.operations.tuners.append(tune)

        while not compress.complete and self.simulation.timestep < 1e6:
            self.simulation.run(1000)

        if not compress.complete:
            message = "Compression failed to complete"
            raise RuntimeError(message)

        print(f"压缩体系完成，{len(positions)} 个有序排列粒子,适宜的移动步长为{self.mc.d['B']},适宜的旋转步长为{self.mc.a['B']}")
    
    def randomizing_particles(self):
        """
        将有序的粒子打乱
        """
        #初始化模拟
        #self.simulation = hoomd.Simulation(device=self.device,seed=1)
        #self.simulation.operations.integrator = self.mc
        #self.simulation.create_state_from_snapshot(self.snapshot)
        #initial_snapshot = self.simulation.state.get_snapshot()

        self.simulation.run(self.pre_random)
        #self.snapshot = self.simulation.state.get_snapshot()

    def save_to_gsd(self):
        logger = hoomd.logging.Logger()
        logger.add(self.mc, quantities=['type_shapes'])
        if self.shape=='A':
            gsd_writer = hoomd.write.GSD(filename='./gsd/P_{:.2f}_{}.gsd'.format(self.packing_density_0,self.num),
                                        trigger=hoomd.trigger.Periodic(100),
                                        mode='wb',filter=hoomd.filter.All(),
                                        logger=logger)
        elif self.shape=='B':
            gsd_writer = hoomd.write.GSD(filename='./gsd_concave/P_concave_{:.2f}_{}.gsd'.format(self.packing_density_0,self.num),
                                        trigger=hoomd.trigger.Periodic(100),
                                        mode='wb',filter=hoomd.filter.All(),
                                        logger=logger)
        self.simulation.operations.writers.append(gsd_writer)
        
    def save_to_gsd_sdf(self):
        logger = hoomd.logging.Logger()
        logger.add(self.mc, quantities=['type_shapes'])
        if self.shape=='A':
            gsd_writer = hoomd.write.GSD(filename='./gsd/P_{:.2f}_{}.gsd'.format(self.packing_density_0,self.num),
                                        trigger=hoomd.trigger.Periodic(100),
                                        mode='wb',filter=hoomd.filter.All(),
                                        logger=logger)
        elif self.shape=='B':
            gsd_writer = hoomd.write.GSD(filename='gsd_sdf/concave/P_concave_{:.2f}_{}_{}.gsd'.format(self.packing_density_0,self.num,self.sdf_file_num),
                                        trigger=hoomd.trigger.Periodic(100),
                                        mode='wb',filter=hoomd.filter.All(),
                                        logger=logger)
        self.simulation.operations.writers.append(gsd_writer)

    def random_inserting(self,insert_times):
        #设置新积分器
        self.new_mc = self.mc
        self.new_mc.shape["A"] = self.mc.shape["A"]

        # 将积分器添加到模拟中
        self.simulation.operations.integrator = self.mc
        
        # 获取模拟盒尺寸
        box = self.simulation.state.box
        Lx = box.Lx
        Ly = box.Ly

        # 获取粒子类型的索引
        type_id = self.simulation.state.particle_types.index('A')

        # 开始插入粒子
        self.inserted_recorder=0
        attempts = 0
        self.old_snap = self.simulation.state.get_snapshot()

        # 记录旧的粒子数量
        N_old = self.old_snap.particles.N
        # 新的粒子数量
        N_new = N_old + 1
        theta = np.random.uniform(0, 2*np.pi)
        
        # 创建一个新的快照，具有更多的粒子
        new_snap = hoomd.Snapshot()
        new_snap.particles.N = N_new
        
        # 复制盒子尺寸
        new_snap.configuration.box = self.old_snap.configuration.box
        
        # 初始化粒子类型
        new_snap.particles.types = self.old_snap.particles.types

        # 初始化属性数组
        new_snap.particles.position[:] = np.zeros((N_new, 3), dtype=float)
        new_snap.particles.orientation[:] = np.zeros((N_new,4), dtype=float)
        new_snap.particles.typeid[:N_old] = self.old_snap.particles.typeid[:]

        # 复制旧的粒子数据
        new_snap.particles.position[:N_old] = self.old_snap.particles.position[:]
        new_snap.particles.orientation[:N_old] = self.old_snap.particles.orientation[:]
        new_snap.particles.typeid[:N_old] = self.old_snap.particles.typeid[:]

        while attempts < insert_times:
            attempts += 1

            x = np.random.uniform(-Lx/2, Lx/2)
            y = np.random.uniform(-Ly/2, Ly/2)

            # 设置新粒子的属性
            new_snap.particles.position[N_old] = [x, y,0]
            new_snap.particles.orientation[N_old] = [np.cos(theta/2), 0, 0, np.sin(theta/2)]
            new_snap.particles.typeid[N_old] = 0

            new_simulation = hoomd.Simulation(device=self.device,seed=1)
            new_simulation.create_state_from_snapshot(new_snap)
            new_simulation.operations.integrator = self.new_mc
            #check_snapshot=simulation.state.get_snapshot()
            #render(check_snapshot)

            # 检查重叠
            new_simulation.run(0)
            if self.new_mc.overlaps == 0:
                #print(f"检测到重叠，移除粒子 {inserted}")
                self.inserted_recorder += 1
        return self.inserted_recorder

    def random_insert(self,insert_times):
        self.fv=hoomd.hpmc.compute.FreeVolume(test_particle_type=self.shape, num_samples=insert_times)
        self.simulation.operations.computes.append(self.fv)
        self.success_insert = round(self.fv.free_volume * insert_times / self.volume)
        return self.success_insert

    def calculate_sdf(self,sdf_mc,sdf_xmax,sdf_dx,sdf_each_run):
        self.total_sdf_sdfcompression=np.zeros(int(sdf_xmax/sdf_dx))
        self.total_sdf_sdfexpansion=np.zeros(int(sdf_xmax/sdf_dx))
        self.sdf_compute = hoomd.hpmc.compute.SDF(xmax=sdf_xmax, dx=sdf_dx)
        self.simulation.operations.computes.append(self.sdf_compute)
        sdf_start_time=time.time()
        print("sdf循环开始")
        for i in range(sdf_mc):
            self.simulation.run(sdf_each_run)
            self.total_sdf_sdfcompression += self.sdf_compute.sdf_compression
            self.total_sdf_sdfexpansion += self.sdf_compute.sdf_expansion
            if (i+1)%(sdf_mc//10)==0:
                sdf_interval_time=time.time()
                print(f"循环已经进行了{i+1}次,耗时{sdf_interval_time-sdf_start_time:.2f}秒")
        self.total_sdf_xcompression = self.sdf_compute.x_compression
        self.total_sdf_sdfcompression /= sdf_mc
        self.total_sdf_sdfexpansion /= sdf_mc
        return self.total_sdf_xcompression,self.total_sdf_sdfcompression,self.total_sdf_sdfexpansion

