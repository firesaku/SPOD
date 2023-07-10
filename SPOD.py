import numpy as np
import random
from sklearn.neighbors import KDTree
import dataPreprocessing
from sklearn.datasets import make_blobs
import math
import copy
import skdim


class Point:
    def __init__(self, order, attribute=None, density=0, detected_time=0, occupied=False, host=None, viscosity=0):
        '''
        点的属性
        :param order:点序号
        :param attribute:点属性（位置）
        :param density:点密度
        :param detected_time:表示点是否被吃的次数
        :param occupied:表示点是否被蜘蛛占领
        :param host:表示占有该点的蜘蛛
        :param viscosity: 网的黏性
        '''
        self.order = order
        self.attribute = attribute
        self.density = density
        self.detected_time = detected_time
        self.occupied = occupied
        self.host = host
        self.viscosity = viscosity


class Spider:
    def __init__(self, order=0, position=None, cost_performance=[], env_value=0, action_module=0, energy=None,
                 threshold=None, view_fild=0, web_range=0, food_list=[], competitive_value=0):
        '''
        蜘蛛的属性
        :param order: 蜘蛛序号
        :param position: 当前占据的点
        :param cost_performance: 不断更新的性价比列表
        :param env_value: 当前所处环境价值
        :param action_module: 行动模式。0为默认，1为飞航，2为捕食，3为网变化
        :param energy: 蜘蛛的能量
        :param threshold: 连续型阈值——高满足->生存
        :param view_fild: 蜘蛛视野范围
        :param web_range: 结网的半径
        :param food_list: 存储蜘蛛的食物
        :param competitive_value: 蜘蛛在竞争阶段各自的值
        '''
        self.order = order
        self.position = position
        self.cost_performance = cost_performance
        self.env_value = env_value
        self.action_module = action_module
        self.energy = energy
        self.threshold = threshold
        self.view_fild = view_fild
        self.web_range = web_range
        self.food_list = food_list
        self.competitive_value = competitive_value


# 点初始化模块
def Init_point(X,view_fild,tree,distances,indices):
    '''
    这个函数用来进行对点的初始化，也就是输入点数据及其属性
    使用方法：point_list=S2.Init_point(data) 这里的data指存有数据文件，视具体情况而填写即可
    :param X: 传入的所有数据属性，比如x,y,z...
    :param view_fild: 蜘蛛视野范围，在此处用于计算点密度
    :param tree: 近邻树
    :param distances: 近邻树生成的距离矩阵
    :param indices: 近邻树生成的近邻矩阵
    :return:  返回存有点对象的属性的列表
    '''
    point_list = []  # 创建空列表
    # 对每个点的属性进行存储
    for i in range(0, X.shape[0]):  # shape[0]表示矩阵的行数，shape[1]表示矩阵的列数
        sum_distance = 0
        sum_k = 0
        if distances[i][1] > view_fild:   # 检查第i个点与其第一个最近邻点之间的距离是否大于`view_fild`
            now_rho = 0
            temp_point = Point(i, attribute=np.array([X[i]]), density=now_rho, detected_time=0, occupied=False)  # 传入属性
            # 创建一个新的数据点对象，该点的索引为i，属性为X[i]，密度为now_rho，检测时间为0，占据状态为False
            point_list.append(temp_point)  # 将temp_point添加到point_list列表中
            continue    # 跳过本次循环的剩余部分，开始下一次循环
        else:   # 如果第i个点与其第一个最近邻点之间的距离小于或等于view_fild
            for j in range(1, len(indices)):     # 遍历该点的其他最近邻点
                if distances[i][j] <= view_fild:      # 检查第i个点与第j个最近邻点之间的距离是否小于或等于view_fild
                    sum_distance += distances[i][j]   # 将距离累加到sum_distance中
                    sum_k += 1    # 计数器sum_k加1
                else:
                    # print(sum_k,distances[i][j])
                    break   # 跳出内部循环
            # TODO 密度值
            # now_rho = sum_k/sum_distance  # 计算密度值now_rho
            now_rho = sum_distance/sum_k  # 计算密度值now_rho
            temp_point = Point(i, attribute=np.array([X[i]]), density=now_rho, detected_time=0, occupied=False)  # 传入属性
            # 创建一个新的数据点对象，该点的索引为i，属性为X[i]，密度为now_rho，检测时间为0，占据状态为False
            point_list.append(temp_point)  # 将temp_point添加到point_list列表中

    return point_list, tree  # 返回存有数据及属性的点列表; tree存储了近邻信息，用于后续捕食范围查询


# Pak
def find_max_k(data, Dthr = 23.928):
    """
    自适应分配每个点对应的k近邻
    :param data: 输入数据
    :param Dthr: 判断阈值。由于密度差符合卡方分布，Dthr越大判断结果越符合点i所拥有的最大近邻。Dtrh对应置信度，因此该条件不是自由变量
    :return: 返回每个点对应的k近邻 密度rou 预测误差error 亮度light 近邻距离矩阵distances 近邻矩阵indices
    """
    # 计算点的内在维度
    TLE = skdim.id.TLE().fit(data)
    id = TLE.dimension_

    # 构建KD树
    # 计算每个数据点与其最近邻之间的距离矩阵（distances）和近邻索引矩阵（indices）
    tree = KDTree(data)
    distances, indices = tree.query(data, k=data.shape[0])      # data.shape[0]——>数据点的总数
    # 计算距离矩阵的幂，将距离矩阵中的每个元素都进行幂运算，幂的指数为内在维度。
    # 目的：对距离进行加权，以突出更大的距离值，主要用于计算密度差
    dissimilarity = np.power(distances, id)
    # 计算距离矩阵的差异矩阵，即密度差矩阵；将距离矩阵的每一行相邻元素之间的差异计算出来
    V_matrix = np.diff(dissimilarity, axis=1)   # axis=1表示沿着行的方向进行计算

    # 初始化每个点对应的k, 密度，预测误差，亮度
    list_k = [-1] * len(data)
    list_rou = [-1] * len(data)
    list_error = [-1] * len(data)
    list_density = [-1] * len(data)

    point_list = []
    for i in range(len(data)):  # 遍历每一个点
        Dk_flag = False     # 设置一个标志变量，判断是否有点满足密度差条件
        now_k = 0   # 当前近邻数
        while True:     # 进入一个无限循环，用于计算满足条件的最大近邻数。
            now_k += 1  # 将当前近邻数增加1，表示计算下一个近邻。
            # 计算now_k
            j = indices[i][now_k]   # 找到当前点的第k个近邻； indices[i][0]表示数据点i本身
            # 计算Dk 和 Dk1 根据给定的公式，计算当前近邻数下的密度差值
            Dk = -2 * now_k * (np.log(np.sum(V_matrix[i][:now_k])) + np.log(np.sum(V_matrix[j][:now_k])) - 2 * np.log(
                np.sum(V_matrix[i][:now_k]) + np.sum(V_matrix[j][:now_k])) + np.log(4))
            Dk1 = -2 * now_k * (np.log(np.sum(V_matrix[i][:now_k+1])) + np.log(np.sum(V_matrix[j][:now_k+1])) - 2 * np.log(
                np.sum(V_matrix[i][:now_k+1]) + np.sum(V_matrix[j][:now_k+1])) + np.log(4))
            # print(Dk,Dk1)
            if Dk < Dthr:     # 判断是否达到密度差的阈值
                Dk_flag = True      # 如果满足条件，将标志变量设置为True。

            # 判断是否达到密度差阈值或已经遍历到最大近邻数，如果【达到阈值】 或者 【遍历到最大近邻数】 则停止遍历
            if ((Dk1 >= Dthr) and (Dk_flag == True)) or (now_k == data.shape[0]-1) == True:
                list_k[i] = now_k   # 将当前数据点的最大近邻数赋值给list_k列表
                list_rou[i] = now_k / np.sum(V_matrix[i][:now_k])   # 计算当前点的密度——>当前近邻数/距离矩阵中前now_k个距离的和
                list_error[i] = np.sqrt((4 * now_k + 2) / ((now_k - 1) * now_k))    # 计算当前数据点的预测误差（error值）
                list_density[i] = list_rou[i] - list_error[i]   # 计算当前数据点的密度，通过将密度值减去预测误差值得到
                # TODO 密度
                # 创建一个Point对象，表示当前数据点，并将其最终的密度属性值（密度减去预测误差）添加到point_list列表中。
                point_list.append(Point(i, attribute=np.array([data[i]]), density=list_rou[i] - list_error[i], detected_time=0, occupied=False))
                break

    return point_list, tree, list_density, list_k, indices, distances


# 蜘蛛初始化模块
def Init_spider(energy, beta, list_light, list_k, indices, distances):
    '''
    这个函数用来进行对点的初始化，也就是输入点数据及其属性
    使用方法：spider_list=S2.Init_spider(data) 这里的data指存有数据文件，视具体情况而填写即可
    :param point_list: 已知的点列表，根据点列表得到蜘蛛列表
    :param spider_num: 蜘蛛的数量
    :param view_fild: 蜘蛛的视野范围
    :param tree: 近邻树
    :param energy: 蜘蛛的能量
    :param beta: 控制蜘蛛的满足度
    :return: 返回存有蜘蛛对象的属性的列表
    '''
    spider = []     # 创建空列表
    # 对每个点的属性进行存储
    for i in range(0, len(list_light)):
        neighbor_num = list_k[i]
        cluster_center_flag = True
        now_env_value = 0
        for j in range(1, neighbor_num):
            now_env_value += list_light[i]
            target_point = indices[i][j]
            if list_light[i] < list_light[target_point]:
                cluster_center_flag = False
                break
        now_env_value = now_env_value/neighbor_num
        if cluster_center_flag==True:
            spider.append(Spider(order=i, position=target_point, env_value=now_env_value, energy=energy,
                             threshold=energy*beta, view_fild=distances[i][j], web_range=distances[i][j]))

    return spider   # 返回存有数据及属性的蜘蛛列表


# 捕食模块
def Predation_Module(X, spider,tree,point_list,alpha,gamma):
    """
    捕食模块
    :param spider:当前蜘蛛
    :param tree: 近邻树
    :param point_list:点列表
    :param alpha: 控制网变化程度
    :param gamma: 控制蜘蛛能量变化程度
    :return:
    """
    spider.action_module = 2    # 标记当前为捕食模式
    spider.web_range = alpha*spider.web_range+(1-alpha)*spider.energy
    # 更新蜘蛛的网大小，通过将当前网大小与新计算得到的值进行加权平均
    food_indices = tree.query_radius(point_list[spider.position].attribute, spider.web_range)[0]
    # 在蜘蛛的位置周围的网范围内查找食物，返回食物所在的数据点序号列表
    new_energy = 0    # 可以获得的能量
    for i in range(0,len(food_indices)):
        order = food_indices[i]
        point_list[order].detected_time += 1  # 表示食物被探测了1次

        # TODO 网的黏性计算
        spider_position = point_list[spider.position].attribute
        point_attribute = np.array([X[order]])
        # 计算距离
        distance = calculate_distance(spider_position, point_attribute)
        # 获取蜘蛛的能量值，蜘蛛的密度进行计算,并对能量、密度取log处理
        # TODO 修改
        now_viscosity = spider.web_range / (1 + distance)
        # now_viscosity = spider.web_range / (1+distance) * spider.energy
        point_list[order].viscosity = alpha*point_list[order].viscosity+now_viscosity * (1 - alpha)

        # 判断是否需要竞争
        if point_list[order].occupied == False:   # 不用竞争
            spider.food_list.append(point_list[order])  # 把当前点加入食物列表
            point_list[order].occupied = True     # 表示食物已经被占据
            point_list[order].host = spider   # 更新食物的拥有者
            new_energy += point_list[order].density   # 更新可获得能量
        else:
            # TODO 竞争值的计算
            calculate_list = []
            # 计算竞争值
            host_spider_position = point_list[point_list[order].host.position].attribute
            host_distance = calculate_distance(host_spider_position, point_attribute)
            spider.competitive_value = spider.energy / distance

            point_list[order].host.competitive_value = point_list[order].host.energy / (1+host_distance)
            calculate_list.append(spider.competitive_value)
            calculate_list.append(point_list[order].host.competitive_value)
            if spider.competitive_value > point_list[order].host.competitive_value:    # 竞争成功，则抢夺食物
                point_list[order].host = spider   # 更改食物的宿主
                spider.food_list.append(point_list[order])  # 添加食物
                point_list[order].host.food_list.remove(point_list[order])  # 更改拥有的食物
                new_energy += point_list[order].density     # 更新可获得能量
    # 能量更新
    new_energy = new_energy
    spider.food_list = list(set(spider.food_list))
    old_threshold = copy.copy(spider.energy)*copy.copy(spider.threshold)
    spider.energy = spider.energy * gamma + new_energy * (1 - gamma)  # 更新能量
    if spider.energy < old_threshold and (new_energy < spider.energy*(gamma)):   # 能量小于阈值 且 从当前环境的获得的能量不如损耗的能量，则飞走
        Flying_Wandering(spider, point_list, gamma)


# 飞航模块
def Flying_Wandering(spider, point_list, gamma):
    print("蜘蛛",spider.order,"飞航")
    now_env_value = 0  # 记录当前环境价值
    distance=0
    for point in point_list:    # 遍历每一个点
        if point.occupied == False:   # 找到未被占领的点
            distance=calculate_distance(point_list[spider.position].attribute,point.attribute)
            spider.position = point.order   # 更新蜘蛛位置
            spider.action_module = 1  # 更新蜘蛛行为模式

            break   # 找到了未被占领的点即停止

    old_energy=spider.energy
    spider.energy = spider.energy-gamma*(now_env_value-spider.env_value)    # 更新能量。如果新环境比旧环境好，则增加能量。不好，则减少能量
    print("飞航前能量为", old_energy, "飞航后能量为", spider.energy)
    for now_food in spider.food_list:  # 对之前的食物作调整
        now_food.host = None  # 更改食物拥有者
        now_food.occupied = False     # 更改食物状态
    spider.food_list = []     # 清空食物列表

    if spider.energy <= 0:    # 如果飞航后蜘蛛死亡，则修改行为
        print("飞航后能量为",spider.energy)
        spider.action_module = -1


# 距离计算
def calculate_distance(spider, point):
    d = 0
    for i in range(spider.shape[1]):
        d += abs((point[0][i] - spider[0][i]) ** 2)

    distance = math.sqrt(d)
    if distance == 0:
        distance = np.random.randint(1,10)*1e-3
    return distance



# 主模块
def Model(X, T, energy, alpha, beta, gamma, contamination, Dthr):
    """
    :param X: 数据集
    :param T: 迭代轮数
    :param energy: 蜘蛛起始能量
    :param alpha: 蛛网变化速率
    :param beta: 控制蜘蛛的满足度阈值
    :param gamma: 蜘蛛能量变化速率
    :param Dthr: 参数越大近邻越大
    :return:
    """
    # 1. 初始化
    # 对于点：要获得对应密度信息
    point_list, tree, list_light, list_k, indices, distances = find_max_k(X, Dthr=Dthr)    # 点初始化

    # 对于蜘蛛：要获得初始能量等信息
    spider_list = Init_spider(energy, beta, list_light, list_k, indices, distances)  # 蜘蛛初始化

    # 2. 开始迭代
    for t in range(0, T):
        print("-----第",t,"轮-----")
        for i in range(0,len(spider_list)):
            # 蜘蛛只有吃或者飞航两种 第一种行为，网改变、捕食竞争属于第二行为。网改变可以在吃完东西后进行，捕食竞争可以在吃东西时进行。
            if spider_list[i].action_module == -1:    # 跳过已经死亡的蜘蛛
                print(spider_list[i].order,"蜘蛛已死")
                continue
            # 蜘蛛刚到了一个新地方【飞航】【初始蜘蛛】
            if spider_list[i].action_module == 0 or spider_list[i].action_module == 1:  # 如果当前spider的行动模块(action_module)为0或1
                print("蜘蛛",spider_list[i].order,"环价",spider_list[i].env_value,"阈值",spider_list[i].energy*spider_list[i].threshold)
                if spider_list[i].env_value > spider_list[i].threshold:     # 如果当前spider的环境值(env_value)大于阈值(threshold)
                    Predation_Module(X, spider_list[i], tree, point_list, alpha, gamma)    # 调用Predation_Module函数，对该spider进行捕食处理
                else:   # 环境价值不够，则飞航
                    Flying_Wandering(spider_list[i], point_list,gamma)  # 调用Flying_Wandering函数，对该spider进行飞航处理
            else:   # 蜘蛛在原位置【捕食】
                print("蜘蛛",spider_list[i].order,"捕食前","能量",spider_list[i].energy)
                Predation_Module(X, spider_list[i], tree, point_list, alpha, gamma)
                print("蜘蛛",spider_list[i].order,"捕食后","能量",spider_list[i].energy)

    occupied_time_list = []
    viscosity_list = []
    rho_list = []
    outlier_list = []
    for point in point_list:
        occupied_time_list.append(point.detected_time)
        viscosity_list.append(point.viscosity*point.detected_time)
        rho_list.append(point.density)

        # 得分
        outlier_list.append(np.log(point.viscosity)*point.detected_time+point.density)

    # TODO 决策图
    # dataPreprocessing.DecisionPlot(rho_list, viscosity_list)
    # 三维决策
    # dataPreprocessing.plot_3d_coordinates(outlier_tube_list, outlier_mucus_list, outlier_passed_time_list)

    indices = [i for i, _ in sorted(enumerate(outlier_list), key=lambda x: x[1])]

    y_predict_type = [0] * len(point_list)
    outlier_num = int(len(point_list) * contamination)
    for i in range(outlier_num):
        y_predict_type[indices[i]] = 1

    return outlier_list, y_predict_type, indices

