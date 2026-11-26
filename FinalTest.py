import numpy as np

from opfunu.cec_based.cec2010 import F42010,F92010,F142010,F82010

from DGEntrance import DifferentialGroupingCalculator
from DataCollectorAndChartPrinter import DataCollector
from CCDEFrame import CCDE

DIM = 1000                # 问题维度
MAX_FES = 3000000    # 最大评估次数
SEED = 42                 # 随机种子
FUNC_CLASS =[F42010,F82010,F92010,F142010]      # 测试函数，可替换为 F22017、F52017 等
DE_PARAMS = {             # DE算法参数
    "NP": 10 * DIM,
    "F": 0.5,
    "CR": 0.9
}

#cec2010的函数，维度固定在1000，有一个m变量表示不可分子组中的变量个数，默认为50，且上下界为±5，和cec2017构造方式有所区别。

if __name__ == "__main__":



    #for i in range(4):
        # 1. 创建测试函数对象
        func = FUNC_CLASS[3](ndim=DIM)

        func_name = func.__class__.__name__
        print(f"测试函数: {func_name}, 维度: {DIM}, 全局最优值: {func.f_global:.6e}")
        results = []
        bounds = np.tile([[-100, 100]], (DIM, 1))
        for j in range(5):
            grouper=DifferentialGroupingCalculator(DIM, bounds,func)
            allgroups,prefes=grouper.getgrouping()

            # 2. 创建数据收集器
            collector = DataCollector(func_name=func_name, dim=DIM, algorithm="SaNSDE/CCDE/DG")

            # 3. 创建DE算法实例
            frame=CCDE(
                DIM,
                bounds,
                50,
                10,
                allgroups,
                prefes,
                5,
                func,
                SEED+j,
                3000000,
            )


            # 4. 运行算法
            print("算法运行中...")
            result,fescost = frame.run()
            results.append(frame.global_best_f)
            av=np.mean(results)
            print(f"运行30次平均最优值水平为：{av:.6e}")