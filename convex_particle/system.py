import math
import hoomd
import numpy as np
import itertools
import time
import gsd.hoomd
import matplotlib.pyplot as plt

#设置使用的设备
cpu = hoomd.device.CPU()

def render(snapshot, dims=2):
    """
    从gsd文件生成一个二维的粒子群图像,目前仅支持圆粒子图.
    """
    # Extract particle positions and convert to a NumPy array
    positions = np.asarray(snapshot.particles.position)

    # Check dimensions (only 2D is supported in this function)
    if dims != 2:
        print("Only 2D rendering is supported in this function.")
        return

    # Plot particles
    plt.figure(figsize=(8, 8))
    plt.scatter(positions[:, 0], positions[:, 1], s=10, alpha=0.6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Particle positions from GSD snapshot')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def project_polygon(vertices, axis):
    """将多边形的顶点投影到给定轴上，返回投影的最小值和最大值。"""
    projections = np.dot(vertices, axis)
    return np.min(projections), np.max(projections)

def is_overlap(polygon1, polygon2):
    """使用分离轴定理检查两个凸多边形是否重叠。"""
    # 获取两个多边形的所有边缘
    edges1 = [(polygon1[i] - polygon1[i - 1]) for i in range(len(polygon1))]
    edges2 = [(polygon2[i] - polygon2[i - 1]) for i in range(len(polygon2))]
    
    # 遍历所有可能的分离轴（即所有边缘的法向量）
    for edge in edges1 + edges2:
        # 计算法向量作为分离轴
        axis = np.array([-edge[1], edge[0]])  # 法向量
        
        # 将两个多边形的顶点投影到该轴上
        min1, max1 = project_polygon(polygon1, axis)
        min2, max2 = project_polygon(polygon2, axis)
        
        # 如果在该轴上没有重叠，则返回 False
        if max1 < min2 or max2 < min1:
            return False
    
    # 如果所有轴上都有重叠，则返回 True
    return True

    # 示例：检测两个三角形是否重叠
    #polygon1 = np.array([[0, 0], [1, 0], [0.5, 1]])
    #polygon2 = np.array([[0.5, 0.5], [1.5, 0.5], [1, 1.5]])

    #if is_overlap(polygon1, polygon2):
        #print("两个多边形重叠")
    #else:
        #print("两个多边形不重叠")

def generate_particle(N_particles):
    #生成一个有N_particles的有序粒子群
    spacing = 2.5
    K = math.ceil(N_particles ** (1 / 2))
    L = K * spacing
    x = np.linspace(-L / 2, L / 2, K, endpoint=False)
    position_2d = list(itertools.product(x, repeat=2))  # 生成二维网格上的粒子位置

    # 取前 N_particles 个位置
    positions_2d = np.array(position_2d[0:N_particles])  # 形状为 (N_particles, 2)

    # 添加第三个坐标（z 坐标），设为零
    z_coordinates = np.zeros((N_particles, 1))  # 形状为 (N_particles, 1)
    positions_3d = np.hstack((positions_2d, z_coordinates))  # 合并为 (N_particles, 3)

    # 创建 GSD 帧并设置粒子属性
    frame = gsd.hoomd.Frame()
    frame.particles.N = N_particles
    frame.particles.position = positions_3d  # 现在是 (N_particles, 3) 的数组
    frame.particles.typeid = [0] * N_particles
    frame.configuration.box = [L, L, 0, 0, 0, 0]  # 注意这里的盒子尺寸

    #创建一个快照
    snapshot=hoomd.Snapshot()
    snapshot.particles.N = N_particles
    snapshot.particles.position[:] = positions_3d
    snapshot.particles.typeid[:] = [0] * N_particles
    snapshot.particles.types=['A']
    snapshot.configuration.box=[L, L, 0, 0, 0, 0]


    return snapshot

def randomizing_particles(snapshot,times=10e3,run_divice=cpu):
    simulation = hoomd.Simulation(device=run_divice, seed=29)

    mc = hoomd.hpmc.integrate.ConvexPolygon(default_d=0.1,default_a=0.2)
    mc.shape["A"] = dict(
        vertices = [
        (-0.5, -0.5),
        (0.5, -0.5),
        (0, 0.5),
    ]
    )

    simulation.operations.integrator = mc
    simulation.create_state_from_snapshot(snapshot)
    initial_snapshot = simulation.state.get_snapshot()

    simulation.run(times)
    final_snapshot = simulation.state.get_snapshot()

    return final_snapshot,mc,simulation

def insert_triangle( simulation , mc,times=100,run_divice=cpu):
    """
    在现有的 HOOMD 模拟中随机插入一个三角型粒子，并检测与已有粒子是否重叠并删除，一共插入指定次数。

    参数：
    simulation: hoomd.Simulation 对象，表示现有的模拟。
    num_triangles: 要插入的三角形粒子的数量。
    """

    #设置新积分器
    new_mc = hoomd.hpmc.integrate.ConvexPolygon()
    new_mc.shape["A"] = dict(
        vertices = [
        (-0.5, -0.5),
        (0.5, -0.5),
        (0, 0.5),
    ]
    )


    # 将积分器添加到模拟中
    simulation.operations.integrator = mc
    

    # 获取模拟盒尺寸
    box = simulation.state.box
    Lx = box.Lx
    Ly = box.Ly

    # 获取粒子类型的索引
    type_id = simulation.state.particle_types.index('A')

    # 开始插入粒子
    inserted_recorder=0
    attempts = 0
    old_snap = simulation.state.get_snapshot()

    while attempts < times:
        attempts += 1
        # 随机生成位置
        x = np.random.uniform(-Lx/2, Lx/2)
        y = np.random.uniform(-Ly/2, Ly/2)
        #print(x,y)

        # 插入粒子
        #if old_snap.communicator.rank == 0:

        # 记录旧的粒子数量
        N_old = old_snap.particles.N
        # 新的粒子数量
        N_new = N_old + 1
        theta = np.random.uniform(0, 2*np.pi)
        
        # 创建一个新的快照，具有更多的粒子
        new_snap = hoomd.Snapshot()
        new_snap.particles.N = N_new
        
        # 复制盒子尺寸
        new_snap.configuration.box = old_snap.configuration.box
        
        # 初始化粒子类型
        new_snap.particles.types = old_snap.particles.types
        
        # 初始化属性数组
        new_snap.particles.position[:] = np.zeros((N_new, 3), dtype=float)
        new_snap.particles.orientation[:] = np.zeros((N_new, 4), dtype=float)
        new_snap.particles.typeid[:N_old] = old_snap.particles.typeid[:]
        
        # 复制旧的粒子数据
        new_snap.particles.position[:N_old] = old_snap.particles.position[:]
        new_snap.particles.orientation[:N_old] = old_snap.particles.orientation[:]
        new_snap.particles.typeid[:N_old] = old_snap.particles.typeid[:]
        
        # 设置新粒子的属性
        new_snap.particles.position[N_old] = [x, y, 0]
        new_snap.particles.orientation[N_old] = [np.cos(theta/2), 0, 0, np.sin(theta/2)]
        new_snap.particles.typeid[N_old] = 0

            
        # 如果有其他属性（如 charge、diameter 等），也需要进行同样的处理

        new_simulation = hoomd.Simulation(device=run_divice,seed=1)
        new_simulation.create_state_from_snapshot(new_snap)
        new_simulation.operations.integrator = new_mc
        #check_snapshot=simulation.state.get_snapshot()
        #render(check_snapshot)

        # 检查重叠
        new_simulation.run(0)
        if new_mc.overlaps > 0:
            #print(f"检测到重叠，移除粒子 {inserted}")
            continue
        else:
            inserted_recorder += 1
        
    return inserted_recorder

def moving_average(data, window_size):
    """计算移动平均,window_size 是滑动窗口大小。"""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def run_with_gpu(sequence, pre_random, real_mc_step, insert_times):
    gpu = hoomd.device.GPU()
    simulation = hoomd.Simulation(device=gpu, seed=42)

    success_probabilities = []
    for i in range(sequence):
        snapshot = generate_particle(5 *i + 1)

        success_times = 0
        total_attempts = real_mc_step * insert_times

        for j in range(real_mc_step + pre_random):
            snapshot, mc, simulation = randomizing_particles(snapshot, times=5)  # Adjust the number of steps as needed
            if j >= pre_random:
                inserted = insert_triangle(simulation, mc, times=insert_times)
                success_times += inserted

        lnP_i = math.log(success_times / total_attempts)
        success_probabilities.append(lnP_i)

    return success_probabilities

class System:
    def __init__(self,num,packing_density,packing_density_0,particle_area,mc,condensed_ratio,margin,pre_random=10,
                 device=cpu):
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

    def generate_particle(self):
        #生成一个有N_particles的有序粒子群
        #K = math.ceil(self.num ** (1 / 2))
        L = width = height = math.sqrt(self.num * self.particle_area / self.packing_density)

        xa,ya=self.mc.shape["A"]["vertices"][2]
        xb,yb=self.mc.shape["A"]["vertices"][0]
        xc,yc=self.mc.shape["A"]["vertices"][1]

        particle_width= xc-xb
        particle_height = ya
        x_a = 27/8
        y_a=5/8

        positions=[]
        orientation=[]

        unit_area = width * height / self.num

        # 计算网格间距，确保三角形之间不重叠
        # 使用更紧凑的间距以提高密度
        #row_spacing = self.condensed_ratio * unit_area/(side_length_a + self.margin)    #height1 * 0.75 + margin   紧凑行间距
        part_area=self.particle_area*self.condensed_ratio/self.packing_density
        expand_ratio=math.sqrt(part_area/(particle_width*particle_height))
        col_spacing = particle_width*expand_ratio    # 列间距保持不变
        row_spacing = particle_height*expand_ratio

        # 计算行列数
        cols = round(width / col_spacing)
        rows = round(height / row_spacing)

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
        snapshot.particles.types=['A']
        snapshot.configuration.box=[L, L, 0, 0, 0, 0]
        self.snapshot=snapshot
        self.simulation = hoomd.Simulation(device=self.device,seed=1)
        self.simulation.operations.integrator = self.mc
        self.simulation.create_state_from_snapshot(self.snapshot)
        
    def generate_system(self):
        num_endpoint=len(self.mc.shape["A"]["vertices"])
        endpoints_x=[]
        endpoints_y=[]
        for i in range(num_endpoint):
            x,y=self.mc.shape["A"]["vertices"][i]
            endpoints_x.append(x)
            endpoints_y.append(y)
        col_spacing=(max(endpoints_x)-min(endpoints_x))*1.2
        row_spacing=(max(endpoints_y)-min(endpoints_y))*1.2
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
        snapshot.particles.typeid[:] = [0] * self.num
        snapshot.particles.types=['A']
        snapshot.configuration.box=[width, height, 0, 0, 0, 0]
        self.snapshot=snapshot
        self.simulation = hoomd.Simulation(device=self.device,seed=1)
        self.simulation.operations.integrator = self.mc
        self.simulation.create_state_from_snapshot(self.snapshot)

        #压缩体系

        final_volume=self.simulation.state.N_particles*self.particle_area/self.packing_density
        #inverse_volume_ramp = hoomd.variant.box.InverseVolumeRamp(
        #initial_box=self.simulation.state.box,
        #final_volume=final_volume,
        #t_start=self.simulation.timestep,
        #t_ramp=20_000,
        #)
        #steps = range(0, 40000, 20)
        #y = [inverse_volume_ramp(step)[0] for step in steps]
        #box_resize = hoomd.update.BoxResize(
        #    trigger=hoomd.trigger.Periodic(10),
        #    box=inverse_volume_ramp,
        #)
        #self.simulation.operations.updaters.append(box_resize)
        #
        #self.simulation.run(20001)
        #
        #self.simulation.operations.updaters.remove(box_resize)

        target_box = hoomd.variant.box.InverseVolumeRamp(
            self.simulation.state.box, final_volume, 0, 1_000)
        qc = hoomd.hpmc.update.QuickCompress(100, target_box)
        count=0
        while (
            self.simulation.timestep < target_box.t_ramp + target_box.t_start or
            not qc.complete):
            self.simulation.run(10)
            count+=1
            if count%100==0:
                print(f"循环{count}次")

        print(f"压缩体系完成，{len(positions)} 个有序排列粒子")
        


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
        gsd_writer = hoomd.write.GSD(filename='./gsd/P_{:.2f}_{}.gsd'.format(self.packing_density_0,self.num),
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
        self.fv=hoomd.hpmc.compute.FreeVolume(test_particle_type="A", num_samples=insert_times)
        self.simulation.operations.computes.append(self.fv)
        self.success_insert = round(self.fv.free_volume * insert_times / self.volume)
        return self.success_insert

    def calculate_sdf(self,sdf_mc,sdf_xmax,sdf_dx,sdf_each_run):
        self.total_sdf_sdfcompression=np.zeros(int(sdf_xmax/sdf_dx))
        simulation_start_time = time.time()
        for i in range(sdf_mc):
            self.simulation.run(sdf_each_run)
            self.sdf_compute = hoomd.hpmc.compute.SDF(xmax=sdf_xmax, dx=sdf_dx)
            self.simulation.operations.computes.append(self.sdf_compute)
            self.total_sdf_sdfcompression += self.sdf_compute.sdf_compression
            simulation_interval_time = time.time()
            if i%(sdf_mc//10)==0:
                print(f"循环已经进行了{i}次,耗时{simulation_interval_time-simulation_start_time:.2f}秒")
        self.total_sdf_xcompression = self.sdf_compute.x_compression
        self.total_sdf_sdfcompression /= sdf_mc
        return self.total_sdf_xcompression,self.total_sdf_sdfcompression

