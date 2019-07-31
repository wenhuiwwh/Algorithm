# -*- coding: utf-8 -*-
"""
新安江模型

@author: QiuQi
"""

import numpy as np

'''
参数赋值
XX：模型参数
K:蒸散发能力折算系数
WUM:上层蓄水容量，它包括植物截留量。在植被与土壤很好的流域，约为20mm；在植被与土壤颇差的流域，约为5～6mm。
WLM:下层蓄水容量，可取60～90 mm。
C:深层蒸散发系数。它决定于深根植物占流域面积的比数，同时也与 值有关，此值越大，深层蒸散发越困难。
一般经验，在江南湿润地区 值约为0.15～0.20左右，而在华北半湿润地区则在0.09～0.12左右。
WM:流域蓄水容量，是流域干旱程度的指标。找久旱以后下大雨的资料，如雨前可认为蓄水量为0，雨后可认为已蓄满，
则此次洪水的总损失量就是 ，可从实测资料中求得，如找不到这样的资料，则只能找久旱以后几次降雨，使雨后蓄满，
用估计的方法求出 。一般分为上层 、下层 和深层 。在南方约为120mm，北方半湿润地区约为180mm。
B:蓄水容量曲线的方次。它反映流域上蓄水容量分布的不均匀性。一般经验，流域越大，各种地质地形配置越多样，
值也越大。在山丘区，很小面积(几平方公里)的B为0.1左右，中等面积(300平方公里以内)的B为0.2～0.3左右，
较大面积(数千平方公里)的B值为0.3～0.4左右。但需说明，B值与WM有关，相互并不完全独立。
同流域同蓄水容量曲线，如WM加大，B就相应减少，或反之。
IMP:不透水面积占全流域面积之比。可找干旱期降小雨的资料来分析，这时有一很小洪水，完全是不透水面积上产生的。
求出此洪水的径流系数，就是IMP。
SM:流域平均自由水蓄水容量，本参数受降雨资料时段均化的影响，当用日为时段长时，一般流域的SM值约为10～50mm。
当所取时段长较少时，SM要加大，这个参数对地面径流的多少起着决定性作用。
EX:自由水蓄水容量曲线指数，它表示自由水容量分布不均匀性。通常EX取值在1~1.5之间。
KSS:自由水蓄水库对壤中流的出流系数
KG:自由水蓄水库对地下径流出流系数
KSS\KG这两个出流系数是并联的，其和代表着自由水出流的快慢。一般来说，KSS+KG=0.7，相当于从雨止到壤中流止的时间为3天。
KKSS:壤中流水库的消退系数。如无深层壤中流时， 趋于零。当深层壤中流很丰富时， 趋于0.9。相当于汇流时间为10天。
KKG:地下水库的消退系数。如以日为时段长，此值一般为0.98~0.998，相当于汇流时间为50~500日。
CS:河网蓄水消退系数，决定于河网地貌
L:滞时，决定于河网地貌
XX0：初值（上一时刻数值）
WU:初始上层蓄水量
WL：初始下层蓄水量
WD：初始深层蓄水量
W：初始蓄水量
FR：初始产流面积
S:初始自由水蓄量
U：流域面积流量换算指数
TRSS0：壤中流初始入流
TRG0：河网初始入流
TR0：滞时总入流
QJ0：初始流量

'''
class parainit:
    
    def __init__(self, XX):
        
        #蒸散发系数
        self.K = XX[0]
        self.WUM = XX[1]
        self.WLM = XX[2]
        self.C = XX[3]
        
        #产流系数
        self.WDM = XX[4]
        self.WM = self.WUM + self.WLM + self.WDM
        self.B = XX[5]
        self.IMP = XX[6]
        
        #分水源系数
        self.SM = XX[7]
        self.EX = XX[8]
        self.KG = XX[9]
        self.KSS = XX[10]
        
        #汇流系数
        self.KKG = XX[11]
        self.KKSS = XX[12]
        self.CS = XX[13]
        self.L = int(np.round(XX[14],0))
        self.U = XX[15] /8.64
                
        #中间值
        self.SMM = (1 + self.EX) * self.SM
        self.WMM = (1 + self.B) * self.WM / (1 - self.IMP) #流域内最大的点蓄水容量
                
    '''赋初值'''    
    def initialvalue(self, XX0):
        
        #初值（上一时刻数值）
        self.WU = XX0[0]
        self.WL = XX0[1]
        self.WD = XX0[2]
        self.W = XX0[3]
        self.FR = XX0[4]
        self.S = XX0[5]
        self.TRSS0 = XX0[6]
        self.TRG0 = XX0[7]
        self.TR0 = XX0[8]
        self.RS = XX0[9]
        self.QJ0 = XX0[10]
    
    #输入降雨量、蒸发量
    def inputv(self, P, EM):
        
        #输入值
        self.P = P
        self.E0 = self.K * EM
        self.PE = self.P - self.E0

# 重新赋值
def deassign( XX0, result):
    
    XX0[0] = result.WU
    XX0[1] = result.WL
    XX0[2] = result.WD
    XX0[3] = result.W
    XX0[4] = result.FR
    XX0[5] = result.S
    XX0[6] = result.TRSS0
    XX0[7] = result.TRG0
    XX0[8] = result.TR0
    XX0[9] = result.RS
    XX0[10] = result.QJ0
    return XX0


'''
产流计算

输出目标-R：产流量

输入参数：
P：降水量
EM：站点蒸发量
E0：流域蒸发量
PE：实际降水量，降水减蒸发
W：当前流域蓄水量
K：蒸散发系数
WM：流域蓄水容量
IMP：不透水面积占全流域面积之比
B：蓄水容量曲线指数
'''
class runoff(parainit):
    
    def depth(self):
        
        #降水小于蒸发，径流R为0
        if self.PE <= 0:
            self.R = 0
            
        else:
            #按照抛物线模型计算
            if self.W >= self.WM:
                A = self.WMM
                
            else:
                A = self.WMM * (1 - (1-self.W/self.WM) ** (1/(1+self.B)))
                
            if A + self.PE > 0:
                if A + self.PE < self.WMM:
                    self.R = self.PE - self.WM + self.W + self.WM * (1-(self.PE + A)/self.WMM) ** (1+self.B)
                else:
                    self.R = self.PE + self.W - self.WM
                    
            else:
                self.R = 0
                
        #print(self.R, self.W)  
'''
蒸散发计算

输出目标：
E蒸发量
EU上层蒸发量、EL下层蒸发量、ED深层蒸发量
W蓄水量
WU上层蓄水量、WL下层蓄水量、WD深层蓄水量

输入参数：
WUM：上层蓄水容量
WLM：下层蓄水容量
C：深层蒸发系数
'''
class evapotranspiration(runoff):
    
    def evcal(self):
        
        #蒸发大于降雨
        if self.PE < 0:
            
            #先蒸发上层
            if self.WU + self.PE > 0:
                
                self.EU = self.E0
                self.ED = 0
                self.EL = 0
                self.WU = self.WU + self.PE
                
            else:
                
                self.EU = self.WU + self.P
                self.WU = 0
                #上层蓄水为0，蒸发下层
                if self.WL > self.C * self.WLM:
                    
                    self.EL = (self.E0 - self.EU) * self.WL / self.WLM
                    self.WL = self.WL - self.EL
                    self.ED = 0
                
                #下层蓄水为0，蒸发深层
                else:
                    
                    if self.WL > self.C * (self.E0 - self.EU):
                        
                        self.EL = self.C * (self.E0 - self.EU)
                        self.WL = self.WL - self.EL
                        self.ED = 0
                        
                    else:
                        
                        self.EL = self.WL
                        self.WL = 0
                        self.ED = self.C * (self.E0 - self.EU) - self.EL
                        self.WD = self.WD - self.ED
                        if self.WD < 0:
                            self.WD = 0
                        
        else:
            #降水大于蒸发
            self.EU = self.E0
            self.ED = 0
            self.EL = 0
            #去除产流是否大于最大点蓄水
            if self.WU + self.PE -self.R < self.WUM:
                #小于
                self.WU = self.WU + self.PE - self.R
                #大于
            else:
                #下渗蓄满
                if self.WU + self.WL + self.PE - self.WUM > self.WLM:
                    
                    self.WU = self.WUM
                    self.WL = self.WLM
                    self.WD = self.W + self.PE - self.R - self.WU - self.WL
                    if self.WD < 0:
                        self.WD = 0
                    
                else:
                    
                    self.WU = self.WUM
                    self.WL = self.WU + self.WL + self.PE - self.R -self.WU
                    
        self.E = self.EU + self.EL + self.ED
        self.W = self.WU + self.WL + self.WD
        

'''
水源划分
先进入自由水蓄量S，再划分水源

此水库有两个出口，一个底孔形成地下径流RG，一个边孔形成壤中流RSS，其出流规律均按线性水库出流分别
成为地下水总入流TRG和壤中流总入流TRSS。由于新安江模型考虑了产流面积FR问题，所以这个自由水蓄水库只发生
在产流面积上，其底宽FR是变化的，产流量R进入水库即在产流面积上，使得自由水蓄水库增加蓄水深，
当自由水蓄水深S超过其最大值SM时，超过部分成为地面径流 。
'''
class components(evapotranspiration):
            
    def comcal(self):
        
        if self.PE <= 0 or self.R <= 0:
            
            self.RS = 0#地表径流
            self.RG = self.S * self.KG * self.FR#地下径流
            self.RSS = self.S * self.KSS * self.FR#壤中流
            
        else:
            
            FR0 = self.FR #本时段产流面积
            self.FR = (self.R - self.PE * self.IMP) / self.PE
            if self.FR > 1:
                self.FR = 1
            if self.FR < 0:
                self.FR = 0.01
            self.S = FR0 * self.S / self.FR#本时段自由蓄水量
            
            SMMF = self.SMM * (1 - (1-self.FR) ** (1/self.EX)) #产流面积上最大一点的自由水蓄水容量
            SMF = SMMF / (1 + self.EX) #产流面积上的平均蓄水容量深
            if self.S >= SMF:
                AU = SMMF
            else:
                AU = SMMF * (1 - (1 - self.S/SMF)) ** (1 / (1 + self.EX))
                
            if AU + self.PE >= SMMF:
                
                self.RS = (self.PE + self.S - SMF) * self.FR + self.RS
                if self.RS < 0:
                    self.RS = 0
                self.RSS = SMF * self.KSS * self.FR
                self.RG = SMF * self.KG * self.FR
                self.S = SMF - (self.RSS + self.RG) / self.FR
                
            else:
                self.RS = (self.PE - SMF + self.S +SMF * (1 - (self.PE + AU) / SMMF)** (1 + self.EX)) * self.FR
                if self.RS < 0:
                    self.RS = 0
                self.RSS = self.KSS * self.FR * (self.PE + self.S - self.RS / self.FR)
                self.RG = self.KG * self.FR * (self.PE + self.S - self.RS / self.FR)
                self.S = self.S + self.PE - (self.RS + self.RSS + self.RG) / self.FR
            #print(self.RG)
                
                
'''
汇流计算
流域汇流计算包括坡地和河网两个汇流阶段。

坡面汇流：
新安江三水源模型中把经过水源划分得到的地面径流RS直接进入河网，成为地面径流对河网的总入流TRS。
壤中流RSS流入壤中流水库，经过壤中流蓄水库的消退（壤中流水库的消退系数为KKSS），成为壤中流对河网总入流TRSS。
地下径流RG进入地下蓄水库，经过地下蓄水库的消退(地下蓄水库的消退系数为KKG)，成为地下水对河网的总入流(TRG)。
河网汇流：
滞后演算法
'''
class confluence(components):
        
    def overlandflow(self):
        
        self.TRS = self.RS * self.U
        self.TRSS = self.TRSS0 * self.KKSS + self.RSS * (1 - self.KKSS) * self.U
        self.TRG = self.TRG0 * self.KKG + self.RG * (1 - self.KKG) * self.U
        self.TR = self.TRS + self.TRSS + self.TRG
        
        #self.QJ = self.CS * self.QJ0 + (1 - self.CS) * self.TR
        
        self.TR0 = self.TR
        self.TRSS0 = self.TRSS
        self.TRG0 = self.TRG
        
def QJcal(QJ,TR, CS, L):
    mQJ = len(QJ)
    for T in range(mQJ - L):
        T1 = T + L
        QJ[T1] =CS*QJ[T-1] + (1 - CS) * TR[T-L]
    return QJ
        
#主函数
def XAJ(xx,xx0,df):
    '''
    xx: 参数
    xx0: 初始值
    df：数据,Dataframe:index-datetime(天),Qr-流量(模型计算不需要,验证结果需要),arearain-面雨量,evag-面蒸发量
    '''    
    result = confluence(xx)
    result.initialvalue(xx0)
    P = df.arearain
    EM = df.evag
    Qs=[]
    TR=[]
    for i in range(len(df)):
        EMi = EM[i]
        Pi = P[i]
        if Pi < 0:
            Pi = 0
        #模型计算
        result.inputv(Pi, EMi)
        result.depth()
        result.evcal()
        result.comcal()
        result.overlandflow()
        Qs.append(result.TR)
        TR.append(result.TR)
    
    return QJcal(Qs,TR, result.CS, result.L)
    
 
    
    
    