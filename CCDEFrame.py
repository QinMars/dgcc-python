import numpy as np
import DGEntrance
from SaNSDEFinal import SaNSDE
from typing import List, Callable
from DataCollectorAndChartPrinter import DataCollector,ChartPrinter

class CCDE:
    def __init__(
        self,
        dim,
        bounds:np.ndarray,
        popsize: int,
        max_iter:int,
        allgroups:List[List[int]],
        FESfromDG:int,
        period:int,
        func,
        seed:int = 42,
        maxfes:int=3000000,
    ):
        self.dim = dim
        self.maxfes=maxfes
        self.fes=FESfromDG
        self.bounds=np.asarray(bounds, dtype=np.float64)
        self.func=func
        self.popsize=popsize
        self.maxiter=max_iter
        self.learningperiod=period
        self.global_best_x: np.ndarray | None = None
        self.global_best_f: float | None = None
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.groupinfo=[]
        self.collector=DataCollector(func.__class__.__name__,dim,"SaNSDE/CCDE/DG")
        self.printer=ChartPrinter((10,6),100)
        print(f"分得组数(一轮cycle的长度):{allgroups.__len__()}")
        self.initial()

        for groupdims in allgroups:
            dimbounds=self.bounds[groupdims,:].copy()
            self.groupinfo.append({"groupdims":groupdims,"dimbounds":dimbounds})
            #这里分维度存储了上下界，不过实际创建的时候，是传入全局上下界创建的，并没有用到，不过可以保留。后续改进算力可以改一下。

        idx = 1
        #在初始化阶段按分组情况创建子求解器，并在类中储存这些子求解器为变量。
        self.solvers = []
        for info in self.groupinfo:
            dim = info["groupdims"]
            bound = info["dimbounds"]
            if len(dim)==0:
                continue
            thefunc = self.maketailorfunc(dim)
            thesolver = SaNSDE(
                self.dim,  # full_dim：全局完整维度
                dim,  # subdim：维度索引列表，例如 [0,2,4]
                thefunc,
                self.popsize,
                self.learningperiod,
                self.maxiter,
                self.pop,
                seed=self.seed + idx,
                bounds=self.bounds
            )
            thesolver.aligngroup(self.pop)
            self.solvers.append(thesolver)
            idx += 1
            #bestx, bestf = thesolver.run()


    def maketailorfunc(self,inputdims)->Callable[[np.ndarray],float]:
        #外层有小组维度传参，然后定制一个函数，先用全局最优解填充，然后把解的目标组维度，填入求解器传入的向量的目标组维度。
        def tailorfunc(x)->float:
            full=self.global_best_x.copy()
            full[inputdims]=x[inputdims]
            self.fes+=1
            fit=self.func.evaluate(full)
            return fit
        return tailorfunc

    def initial(self):
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        self.global_best_x = self.rng.uniform(low, high, size=self.dim)
        self.global_best_f = self.func.evaluate(self.global_best_x)
        self.fes += 1
        self.pop = self.rng.uniform(
            low,
            high,
            size=(self.popsize, self.dim)
        )

        self.fitness = np.array(
            [
                self.func.evaluate(x)
                for x in self.pop
            ]
        )
        self.fes+=self.popsize
        self.best_id = np.argmin(self.fitness)
        self.global_best_x= self.pop[self.best_id].copy()
        self.global_best_f = self.fitness[self.best_id]

        #初始化，主要得到一个随机的解暂时作为全局最优
    def cycle(self):
        #根据保存的groupinfo{维度，上下界}
        idx=1
        for solver in self.solvers:
            print(f"cycle内序号：{idx}")
            idx+=1
            solver.aligngroup(self.pop)
            bestx, bestf = solver.run()
            self.pop=solver.pop.copy()
            #self.fitness=solver.fitness.copy()
            #print(f"当前子求解器得到的最佳适应值为{bestf:.6e}")
            if bestf < self.global_best_f:
                self.global_best_f = bestf
                thecopy = self.global_best_x.copy()
                thecopy[solver.subdim] = bestx[solver.subdim]
                self.global_best_x = thecopy
                print(f"触发全局最优解替换,替换后的最佳适应值为{bestf:.6e}")
            #print(f"当前fes:{self.fes:.6e}")
            self.collector.record(self.fes, self.global_best_f, self.func.f_global)


            if self.global_best_f-self.func.f_global<1e-6:
                print(f"达到最优值，提前终止，fes消耗量：{self.fes}")
                break
            #子优化器得到一个最佳解，提取其关注维度和最佳适应值
            #如果优于全局适应值，就把全局最优解的对应维度填入该优化器得到的结果，并更新全局最优值
    def run(self):

        idx=0
        while self.fes<self.maxfes:
         self.cycle()
         idx+=1
         print(f"while内序号：{idx}")
         print(f"目前fes:{self.fes:.6e}最大值:{self.maxfes:.6e}")
         if self.fes>self.maxfes:
             self.printer.plot(self.collector, 1, 1, 1)
             break
         if self.global_best_f - self.func.f_global < 1e-6:
             self.printer.plot(self.collector, 1, 1, 1)
             break


        return self.global_best_f,self.fes


"""
                    需要接收DGEntrance 差分分组得到的分组结果√
                    然后根据分组结果，创建若干个SaNSDE优化器√
                    需在这里接收各维度的上下界√
                    然后在这里初始化并保存一个全局最优的解√
                    然后优化器变异得到的结果只关注所属的维度，√
                    这里提供一个接口，能够将全局最优解的补集维度粘接给每个优化器√
                    并且在这里传入完整解调用函数并记录FES消耗，所以需要改写func，而不能使用原生√
                    需要在这里统计全局的FES，接受DG传参中的FES√
                    """
