import numpy as np
from CoreDE import DifferentialEvolution
from SaNSDE import SaNSDE

class DifferentialGroupingCalculator:
    def __init__(self,dim,bounds,func):
        self.dims=[]
        self.func=func
        self.fes=0
        for i in range(0,dim):
            self.dims.append(i)#用列表存一个代检测维度
        self.allgroups=[]
        self.seps=[]
        self.standard=0.001#验证标准依据经验
        while self.dims:
            i=self.dims.pop(0)
            current=[i,]
            p=np.full( dim,bounds[:,0])
            q=np.copy(p)
            q[i]=bounds[i,1]
            delta1=self.func.evaluate(q)-self.func.evaluate(p)
            self.fes+=2

            for j in list(self.dims):
                if j==i:
                    continue
                pp=np.copy(p)
                qq=np.copy(q)
                pp[j]=( bounds[j,0]+ bounds[j,1])/2
                qq[j]=( bounds[j,0]+ bounds[j,1])/2
                delta2 = self.func.evaluate(qq) - self.func.evaluate(pp)
                self.fes+=2
                if abs(delta2-delta1)>self.standard:
                    current.append(j)
            for z in current:
                if z in self.dims:
                  self.dims.remove(z)
            if len(current)==1:
                self.seps.append(i)
            else:
                self.allgroups.append(current)
        self.allgroups.append(self.seps)
        #for x in self.seps:
            #self.allgroups.append([x])
        print(self.allgroups)

    def getgrouping(self):
        return self.allgroups,self.fes
