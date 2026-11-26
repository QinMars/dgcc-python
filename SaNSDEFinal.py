import numpy as np


class SaNSDE:
    def __init__(self, dim, subdim, func, popsize, learning_period, max_iter, pop: np.ndarray, seed, bounds):
        # 针对CC框架设计的子求解器。需要只改变部分维度，其余维度不得改变。
        # 每一次运行的时候，都需要接收传入的数组，然后根据现有的全局最优值，拼接（这部分工作交给func即可）并且评估适应度。
        self.dim = dim
        self.subdim = subdim
        self.bounds = bounds  # 记录函数定义域
        self.func = func
        self.popsize = popsize
        self.pop = pop.copy()
        self.fgroup: np.ndarray | None = None
        self.crgroup: np.ndarray | None = None
        self.learning_period = learning_period
        self.max_iter = max_iter
        self.p = 0.5
        self.cr: np.ndarray | float = 0.5  # 后续会被n维随机生成的数组覆盖
        self.crm = 0.5  # 初始设为和cr相等
        self.f = 0.5
        self.fp = 0.5
        self.rng = np.random.default_rng(seed=seed)
        self.global_best_x: np.ndarray | None = None
        self.gen = 0  # 记录代数，以便随学习周期更新参数
        self.s_listener = np.zeros(self.popsize, dtype=int)  # 用该变量监听每次使用的变异策略是什么.s表示strategy策略
        self.f_listener = np.zeros(self.popsize, dtype=int)  # 用该变量监听每次使用的f策略是什么
        self.global_best_f: float | None = None
        """
        self.sns1=0#1表示策略1
        self.sns2=0
        self.snf1=0
        self.snf2=0
        舍弃该表达，用列表便捷表示。
        """
        self.sset = [0, 0, 0, 0]

        # 存为一个列表，就可以通过listener的值快速调用相应的记录。

        # 对于选择策略，和f，各自用四个统计量。
        """
        self.fns1=0
        self.fns2=0
        self.fnf1=0
        self.fnf2=0
        """
        self.fset = [0, 0, 0, 0]

        self.scr = []
        self.imp = []

    def fstep(self):
        # 变异步骤，涉及到p,fp,正态分布和柯西分布
        # 注意，为了搭配cc框架，子优化器只能处理管辖维度
        self.fgroup = self.pop.copy()
        Fs = self.rng.normal(loc=0.5, scale=0.3, size=self.popsize)
        fcauchy = self.rng.standard_cauchy(self.popsize)
        uniforms = self.rng.uniform(0, 1, size=self.popsize)
        uniformsf = self.rng.uniform(0, 1, size=self.popsize)

        for i in range(self.popsize):
            j, k, l = self.get_three_indices(i)
            # 为了保证算法严谨，设置了一个计算函数专门提取排除i的随机数
            if uniforms[i] < self.p:
                # 策略选用1
                self.s_listener[i] = 0
                if uniformsf[i] < self.fp:
                    self.f_listener[i] = 0
                    # f策略选用正态分布
                    self.fgroup[i][self.subdim] = self.pop[j][self.subdim] + Fs[i] * (
                                self.pop[k][self.subdim] - self.pop[l][self.subdim])
                else:
                    self.f_listener[i] = 1
                    # f选用柯西分布
                    self.fgroup[i][self.subdim] = self.pop[j][self.subdim] + fcauchy[i] * (
                                self.pop[k][self.subdim] - self.pop[l][self.subdim])
            else:
                # 策略选用3
                self.s_listener[i] = 1
                if uniformsf[i] < self.fp:
                    self.f_listener[i] = 0
                    # 对于f的监听，不赘述。
                    self.fgroup[i][self.subdim] = self.pop[i][self.subdim] + Fs[i] * (
                                self.global_best_x[self.subdim] - self.pop[i][self.subdim]) + Fs[i] * (
                                                              self.pop[k][self.subdim] - self.pop[l][self.subdim])
                else:
                    self.f_listener[i] = 1
                    self.fgroup[i][self.subdim] = self.pop[i][self.subdim] + fcauchy[i] * (
                                self.global_best_x[self.subdim] - self.pop[i][self.subdim]) + fcauchy[i] * (
                                                              self.pop[k][self.subdim] - self.pop[l][self.subdim])
        self.beyond_case()  # 边界处理
        # 并且还需要对各个策略实行存档的机制，记录选择各策略后，成功和失败的次数。

    def get_three_indices(self, target_idx):
        while True:
            r1, r2, r3 = self.rng.choice(self.popsize, size=3, replace=False)
            if (r1 != target_idx and r2 != target_idx and r3 != target_idx):
                return r1, r2, r3

    def crstep(self):
        # 交叉步骤
        self.crgroup = self.pop.copy()
        self.cr = self.rng.normal(loc=self.crm, scale=0.1, size=self.popsize)
        self.cr = np.clip(self.cr, 0, 1)  # 避免负数或越界
        # cr步骤的杂交方法：对每个个体，生成一个rn（整数）锁定一个维度，然后逐维度生成一个r，根据cr确定是否保留父辈
        locks = self.rng.choice(self.subdim, size=self.popsize)

        for i in range(self.popsize):
            # 这里是否需要仅操作部分维度？感觉并不需要了。其实他们的非管辖维度的确都是一样的。
            rs = self.rng.uniform(0, 1, size=self.dim)
            for j in self.subdim:
                if rs[j] <= self.cr[i] or j == locks[i]:
                    self.crgroup[i][j] = self.fgroup[i][j]
                else:
                    self.crgroup[i][j] = self.pop[i][j]

    def choose(self):
        for i in range(self.popsize):
            popfit = self.fitness[i]
            crfit = self.func(self.crgroup[i])

            if popfit >= crfit:  # 这里就该更新策略的输赢。
                # 这里策略成功了，需要s和f的成功历史分别加1,
                # 并且还需要记录这里的成功参数cr，并记录其带来的适应值改进量。
                self.sset[self.s_listener[i]] += 1
                # s_l为0的时候对应策略1，s_l为1的时候对应策略3,没问题
                self.fset[self.f_listener[i]] += 1

                self.scr.append(self.cr[i])
                self.imp.append(popfit - crfit)

                self.pop[i] = self.crgroup[i]
                self.fitness[i] = crfit
                if crfit < self.global_best_f:
                    self.global_best_f = crfit
                    self.global_best_x = self.pop[i].copy()

            # 如果适应值更好，就更新适应值，否则应该保留旧个体的适应值不变。
            # 然后试图挑战目前最优的适应值和个体位置
            else:
                # 如果没有进入上述if语句，代表策略失败了，需要再s和f的失败历史分别加1
                self.sset[self.s_listener[i] + 2] += 1
                self.fset[self.f_listener[i] + 2] += 1
        self.gen += 1
        if self.gen % self.learning_period == 0:
            self.update_learning()
            # 选择步骤,顺便在这里设立学习周期监测和参数更新的触发

    def cal(self, ns1, ns2, nf1, nf2):
        # 对于策略选择p和F策略选择fp，更新公式是一样的。
        numerator = ns1 * (ns2 + nf2)
        denominator = (ns2 * (ns1 + nf1) + ns1 * (ns2 + nf2))

        if denominator == 0:
            return -1

        return numerator / denominator

        # return ns1*(ns2+nf2)/(ns2*(ns1+nf1)+ns1*(ns2+nf2))

    def update_learning(self):

        if self.cal(*self.sset) != -1:
            self.p = self.cal(*self.sset)

        if self.cal(*self.fset) != -1:
            self.fp = self.cal(*self.fset)

        allimp = sum(self.imp)

        if allimp > 0:  # 如果allimp不是正数，就不要更新了
            self.crm = 0
            for i in range(len(self.scr)):
                self.crm += self.scr[i] * self.imp[i] / allimp
        self.sset = [0, 0, 0, 0]
        self.fset = [0, 0, 0, 0]  # 重新赋值重新统计
        self.scr = []
        self.imp = []

    def beyond_case(self):
        # 采取超出边界的地方重新随机生成值的方式，修复仅发生在fgroup生成的时候
        # bounds的形状要求为(dim,2)

        for i in range(self.popsize):
            for j in self.subdim:
                if self.fgroup[i][j] < self.bounds[j][0] or self.fgroup[i][j] > self.bounds[j][1]:
                    self.fgroup[i][j] = self.rng.uniform(self.bounds[j][0], self.bounds[j][1])

    def aligngroup(self, pop):
        self.pop = pop.copy()
        self.fitness = np.array(
            [
                self.func(x)
                for x in self.pop
            ]
        )
        best_idx = np.argmin(self.fitness)
        self.global_best_x = self.pop[best_idx].copy()
        self.global_best_f = self.fitness[best_idx]

    def cycle(self):
        self.fstep()
        self.crstep()
        self.choose()

    def run(self):
        while (self.gen < self.max_iter):
            self.cycle()
        self.gen = 0
        return self.global_best_x.copy(), self.global_best_f
