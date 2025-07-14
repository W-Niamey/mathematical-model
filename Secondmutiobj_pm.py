import os
import numpy as np
import gurobipy as gp
import pandas as pd
from gurobipy import GRB
import proplem_parser
# from WangYuHang.dpfgsp_wangyuhang_model_ver2 import c_max
from proplem_parser import fields
from pygantt.pygantt import *
import thriftpy2 as thriftpy
import matplotlib
matplotlib.use('TkAgg')
class FGSProblem(proplem_parser.Problem):
    F = fields.IntegerField('Factories')
    G = fields.IntegerField('Families')
    M = fields.IntegerField('Machines')
    J = fields.IntegerField('Total number of jobs')
    SetupType = fields.IntegerField('SetupType')
    J_g = fields.ListField('Number of Jobs in each Family')
    jobs_in_each_family = fields.IndexedCoordinatesField('Jobs in each Family')
    process_time = fields.MatrixField('Processing times of jobs')
    setup_time = fields.MatrixField('Setup times')
    ML = fields.MatrixField('Maximum of maintenance level of machines')
    MT = fields.MatrixField('The maintenance time on machines')

    def __init__(self, **data):
        super().__init__(**data)

class FGSPModel:
    max_infinity = 0x0000ffff

    def __init__(self, filename):
        self.problem = proplem_parser.load(os.path.dirname(os.path.abspath(__file__)) + '/data/' + filename,
                                           problem_class=FGSProblem)
        self.num_of_factories = self.problem.F
        self.factory_array = np.arange(0, self.num_of_factories + 1)

        self.num_of_groups = self.problem.G
        self.group_array = np.arange(0, self.num_of_groups + 1)

        self.num_of_machines = self.problem.M
        self.machine_array = np.arange(0, self.num_of_machines + 1)

        self.num_of_jobs = self.problem.J

        self.num_of_jobs_in_each_group = np.insert(np.array(self.problem.J_g), 0, 0)

        self.jobs_in_each_group = self.problem.jobs_in_each_family

        for value in self.jobs_in_each_group.values():
            value.insert(0, 0)

        self.temp = np.array([2.0])

        self.process_time = np.array(self.problem.process_time)
        self.process_time = np.insert(self.process_time, 0, np.zeros((1, self.num_of_machines), dtype=float), 0)
        self.process_time = np.insert(self.process_time, 0, np.zeros((1, self.num_of_jobs + 1), dtype=float), 1)

        self.setup_time = np.array(self.problem.setup_time)
        self.setup_time = np.insert(self.setup_time, 0, np.zeros((self.num_of_groups, self.num_of_groups), dtype=int), 0)

        for s in range(0, self.num_of_machines + 1):
            self.setup_time = np.insert(self.setup_time, s * (self.num_of_groups + 1),
                                        np.zeros((1, self.num_of_groups), dtype=int), 0)
        self.setup_time = np.insert(self.setup_time, 0,
                                    np.zeros((1, (self.num_of_machines + 1) * (self.num_of_groups + 1)), dtype=int), 1)
        self.setup_time = np.reshape(self.setup_time,
                                     (self.num_of_machines + 1, self.num_of_groups + 1, self.num_of_groups + 1))

        self.speed_array = np.array([0, 1, 2, 3])
        self.v = np.array([0.0, 1.0, 1.0/1.5, 1.0/2.0])
        self.v1 = np.array([0.0, 1.0, 1.5, 2.0])

        self.ml= np.insert(np.array(self.problem.ML), 0, 0)
        self.mt= np.insert(np.array(self.problem.MT), 0, 0)

        self.model = None
        self.w = None
        self.c = None #c
        self.mo  = None # 机器关闭时间
        self.x = None
        self.y = None
        self.z = None
        self.t = None
        self.L = None
        self.c_max = None
        self.TPE = None
        self.TSE = None
        self.TIE = None
        self.TEC = None

    def creat_model(self):
        # Create a new model
        self.model = gp.Model("fgsp_model")
        self.model.setParam(GRB.Param.TimeLimit, 3600)
        self.model.setParam(GRB.Param.IntFeasTol, 1e-9)

        # Create variables
        self.w = self.model.addVars(self.group_array, self.factory_array[1:], vtype=GRB.BINARY, name="w")

        self.c = {}
        for l in self.group_array[1:]:
            self.c[l] = self.model.addVars(self.jobs_in_each_group[l][1:], [l], self.machine_array[1:],
                                           vtype=GRB.CONTINUOUS, name="c")

        self.mo = self.model.addVars(self.machine_array[1:],self.machine_array[1:], vtype=GRB.CONTINUOUS, name="mo")

        self.x = self.model.addVars(self.group_array, self.group_array, self.factory_array[1:], vtype=GRB.BINARY,
                                    name="x")

        self.y = {}
        for l in self.group_array[1:]:
            self.y[l] = self.model.addVars(self.jobs_in_each_group[l], self.jobs_in_each_group[l], [l],
                                           vtype=GRB.BINARY, name="y")
        self.z = {}
        for l in self.group_array[1:]:
            self.z[l] = self.model.addVars(self.jobs_in_each_group[l][1:], [l], self.machine_array[1:],
                                           self.speed_array[1:],
                                           vtype=GRB.BINARY, name="z")

        self.t = {}
        for l in self.group_array[1:]:
            self.t[l] = self.model.addVars(self.jobs_in_each_group[l][1:], [l], self.machine_array[1:],
                                           self.factory_array[1:], vtype=GRB.BINARY, name="t")

        self.L = {}
        for l in self.group_array[1:]:
            self.L[l] = self.model.addVars(self.jobs_in_each_group[l][1:], [l], self.machine_array[1:],
                                           self.factory_array[1:], vtype=GRB.INTEGER, name="L")

        self.c_max = self.model.addVar(vtype=GRB.CONTINUOUS, name='c_max')
        self.TPE = self.model.addVars(self.factory_array[1:], vtype=GRB.CONTINUOUS, name='TPE')
        self.TSE = self.model.addVars(self.factory_array[1:], vtype=GRB.CONTINUOUS, name='TSE')
        self.TIE = self.model.addVars(self.factory_array[1:], vtype=GRB.CONTINUOUS, name='TIE')
        self.TEC = self.model.addVar(vtype=GRB.CONTINUOUS, name='TEC')

        # Add constraints
        self.constrants1()
        self.constrants2()
        self.constrants3()
        self.constrants4()
        self.constrants5()
        self.constrants6()
        self.constrants7()
        self.constrants8()
        self.constrants9()
        self.constrants10()
        self.constrants11()
        self.constrants12()
        self.constrants13()
        self.constrants14()
        self.constrants15()
        self.constrants16()
        self.constrants17()
        self.constrants18()
        self.constrants19()
        self.constrants20()
        self.constrants21()
        self.constrants22()
        self.constrants23()
        self.constrants24()
        self.constrants25()
        self.constrants26()
        self.constrants27()
        self.constrants28()
        self.constrants29()
        self.constrants30()
        # self.constrants31()
        # self.constrants32()


        # Set objective
        # self.model.setObjective(self.c_max, GRB.MINIMIZE)
        self.model.setObjectiveN(self.c_max, 0, priority=2)
        self.model.setObjectiveN(self.TEC, 1, priority=1)
        # self.model.setObjectiveN(self.c_max, index=0, weight=5)
        # self.model.setObjectiveN(self.TEC, index=1, weight=0.5)

        # Optimize model
        self.model.optimize()

        self.print_vars()
        self.graph()

   #约束2 每个组只能安排在一个工厂内
    def constrants1(self):
        self.model.addConstrs(gp.quicksum(self.w[l, f] for f in self.factory_array[1:]) == 1
                              for l in self.group_array[1:])
    #2.3.4.5对应约束3.4.5.6
    def constrants2(self):
        self.model.addConstrs(gp.quicksum(self.x[l, l1, f] for l1 in self.group_array if l1 != l) == self.w[l, f]
                              for l in self.group_array[1:]
                              for f in self.factory_array[1:])

    def constrants3(self):
        self.model.addConstrs(gp.quicksum(self.x[l, l1, f] for l in self.group_array if l != l1) == self.w[l1, f]
                              for l1 in self.group_array[1:]
                              for f in self.factory_array[1:])

    def constrants4(self):
        self.model.addConstrs(gp.quicksum(self.x[0, l1, f] for l1 in self.group_array[1:]) == 1
                              for f in self.factory_array[1:])

    def constrants5(self):
        self.model.addConstrs(gp.quicksum(self.x[l, 0, f] for l in self.group_array[1:]) == 1
                              for f in self.factory_array[1:])
    #6.7对应约束7.8
    def constrants6(self):
        self.model.addConstrs(gp.quicksum(self.y[l][j, j1, l] for j1 in self.jobs_in_each_group[l] if j1 != j) == 1
                              for l in self.group_array[1:]
                              for j in self.jobs_in_each_group[l])

    def constrants7(self):
        self.model.addConstrs(gp.quicksum(self.y[l][j, j1, l] for j in self.jobs_in_each_group[l] if j != j1) == 1
                              for l in self.group_array[1:]
                              for j1 in self.jobs_in_each_group[l])

    #8对应约束9
    def constrants8(self):
        # guarantees that each operation runs at one speed level.
        self.model.addConstrs(gp.quicksum(self.z[l][j, l, i, s] for s in self.speed_array[1:]) == 1
                              for l in self.group_array[1:]
                              for j in self.jobs_in_each_group[l][1:]
                              for i in self.machine_array[1:])

    #9对应约束11
    def constrants9(self):
        # For two jobs within the identical group, if job1 directly succeeds job2 on Mi,
        # the completion time of job1 on Mi is greater than or equal to that of job2 plus pj',l,i
        self.model.addConstrs(
            self.c[l][j1, l, i] >= self.c[l][j, l, i] + self.process_time[j1][i] * gp.quicksum(self.z[l][j1, l, i, s] * self.v[s] for s in self.speed_array[1:]) + self.t[l][j1, l, i, k] * self.mt[i]  + (self.y[l][j, j1, l] - 1) * self.max_infinity
            for l in self.group_array[1:]
            for j in self.jobs_in_each_group[l][1:]
            for j1 in self.jobs_in_each_group[l][1:]
            if j != j1
            for i in self.machine_array[1:]
            for k in self.factory_array[1:]
        )

    #10对应约束12
    def constrants10(self):
        # 为了表达不同组之间工作顺序的关系并加上设置时间、处理时间以及工厂因素
        self.model.addConstrs(
            self.c[l1][j1, l1, i] >= self.c[l][j, l, i] + self.setup_time[i, l, l1] + self.process_time[j1][i] * gp.quicksum(self.z[l1][j1, l1, i, s] * self.v[s] for s in self.speed_array[1:]) +
            self.t[l1][j1, l1, i, k] * self.mt[i] + (self.x[l, l1, k] + self.y[l1][0, j1, l1] + self.y[l][j,0,l] - 3) * self.max_infinity
            for l in self.group_array[1:]  # 遍历每个组
            for l1 in self.group_array[1:]  # 遍历每个其他组
            if l != l1  # 确保 l 和 l1 不相同
            for j in self.jobs_in_each_group[l][1:]  # 遍历组 l 中的作业
            for j1 in self.jobs_in_each_group[l1][1:]  # 遍历组 l1 中的作业
            for i in self.machine_array[1:]  # 遍历所有机器
            for k in self.factory_array[1:]
        )

    #11对应约束13
    def constrants11(self):
        self.model.addConstrs(
            self.c[l][j, l, i] >= self.setup_time[i, l, l] + self.process_time[j][i] * gp.quicksum(self.z[l][j, l, i, s] * self.v[s] for s in self.speed_array[1:]) + self.t[l][j, l, i, k] * self.mt[i] + (self.x[0, l, k] + self.y[l][0, j, l] - 2) * self.max_infinity
            for l in self.group_array[1:]  # 遍历每个组
            for j in self.jobs_in_each_group[l][1:]  # 遍历组 l 中的作业
            for i in self.machine_array[1:]  # 遍历所有机器
            for k in self.factory_array[1:]  # 遍历所有工厂
        )

    #12对应约束14
    def constrants12(self):
        self.model.addConstrs(self.c[l][j, l, i + 1] >= self.c[l][j, l, i] + self.process_time[j][i + 1] * gp.quicksum(
            self.z[l][j, l, i + 1, s] * self.v[s] for s in self.speed_array[1:])
                              for l in self.group_array[1:]
                              for j in self.jobs_in_each_group[l][1:]
                              for i in self.machine_array[1:-1])

    # 13对应约束15 维修约束
    def constrants13 (self):
        self.model.addConstrs(self.L[l][j, l, i, k] >= self.ml[i] -
                              (2 - self.x[0, l, k] - self.y[l][0, j, l]) * self.max_infinity
                              for l in self.group_array[1:]
                              for j in self.jobs_in_each_group[l][1:]
                              for i in self.machine_array[1:]
                              for k in self.factory_array[1:])

    # 14对应约束16
    def constrants14(self):
        self.model.addConstrs(self.L[l][j, l, i, k] <= self.ml[i] +
                              (2 - self.x[0, l, k] - self.y[l][0, j, l]) * self.max_infinity
                              for l in self.group_array[1:]
                              for j in self.jobs_in_each_group[l][1:]
                              for i in self.machine_array[1:]
                              for k in self.factory_array[1:])

    # 15对应约束17
    def constrants15(self):
        self.model.addConstrs(
            self.L[l][j, l, i, k] >= self.process_time[j][i]
            for l in self.group_array[1:]  # 遍历每个组
            for j in self.jobs_in_each_group[l][1:]  # 遍历组 l 中的作业
            for i in self.machine_array[1:]  # 遍历所有机器
            for k in self.factory_array[1:]  # 遍历所有工厂
        )
#28对应约束24
    def constrants28(self):
        self.model.addConstrs(
            1 + self.L[l][j, l, i, k] - self.process_time[j][i] <= self.process_time[j1][i]  + (2 - self.y[l][j, j1, l] - self.t[l][j1, l, i, k]) * self.max_infinity
            for l in self.group_array[1:]  # 遍历每个组
            for j in self.jobs_in_each_group[l][1:]  # 遍历组 l 中的作业
            for j1 in self.jobs_in_each_group[l][1:]
            if j != j1
            for i in self.machine_array[1:]  # 遍历所有机器
            for k in self.factory_array[1:]  # 遍历所有工厂
        )

    # 29对应约束25
    def constrants29(self):
        self.model.addConstrs(
            1 + self.L[l][j, l, i, k] - self.process_time[j][i] <= self.process_time[j1][i] + (4 - self.x[l, l1, k] - self.y[l1][0, j1, l1] - self.y[l][j, 0, l] - self.t[l1][j1, l1, i, k]) * self.max_infinity
            for l in self.group_array[1:]  # 遍历每个组
            for l1 in self.group_array[1:]  # 遍历每个
            if l != l1
            for j in self.jobs_in_each_group[l][1:]  # 遍历组 l 中的作业
            for j1 in self.jobs_in_each_group[l1][1:]
            for i in self.machine_array[1:]  # 遍历所有机器
            for k in self.factory_array[1:]  # 遍历所有工厂
        )

    # 30对应约束26
    def constrants30(self):
        self.model.addConstrs(
            self.t[l][j, l, i, k] <= (2 - self.x[0, l, k] - self.y[l][0, j, l]) * self.max_infinity
            for l in self.group_array[1:]  # 遍历每个组
            for j in self.jobs_in_each_group[l][1:]  # 遍历组 l 中的作业
            for i in self.machine_array[1:]  # 遍历所有机器
            for k in self.factory_array[1:]  # 遍历所有工厂
        )


    # 16对应约束18
    def constrants16(self):
        self.model.addConstrs(
            self.L[l][j1, l, i, k]
            >= self.L[l][j, l, i, k] - self.process_time[j][i] * gp.quicksum(self.z[l][j, l, i, s] * self.v[s] for s in self.speed_array[1:]) -(1 - self.y[l][j, j1, l] + self.t[l][j1, l, i, k]) * self.max_infinity
            for l in self.group_array[1:]
            for j in self.jobs_in_each_group[l][1:]
            for j1 in self.jobs_in_each_group[l][1:]
            if j != j1
            for i in self.machine_array[1:]
            for k in self.factory_array[1:]
        )

    # 17对应约束19
    def constrants17(self):
        self.model.addConstrs(self.L[l][j1, l, i, k] <= self.L[l][j, l, i, k] - self.process_time[j][i] + (1 - self.y[l][j, j1, l] + self.t[l][j1, l, i, k]) * self.max_infinity
                              for l in self.group_array[1:]
                              for j in self.jobs_in_each_group[l][1:]
                              for j1 in self.jobs_in_each_group[l][1:]
                              if j != j1
                              for i in self.machine_array[1:]
                              for k in self.factory_array[1:])

    # 18对应约束20
    def constrants18(self):
        self.model.addConstrs(self.L[l1][j1, l1, i, k] >= self.L[l][j, l, i, k]
                              - self.process_time[j][i] - (3 - self.x[l, l1, k] - self.y[l1][0, j1, l1] - self.y[l][j, 0, l] + self.t[l1][j1, l1, i, k]) * self.max_infinity
                              for l in self.group_array[1:]
                              for l1 in self.group_array[1:]
                              if l != l1
                              for j in self.jobs_in_each_group[l][1:]
                              for j1 in self.jobs_in_each_group[l1][1:]
                              for i in self.machine_array[1:]
                              for k in self.factory_array[1:])

    # 18对应约束21
    def constrants19(self):
        self.model.addConstrs(self.L[l1][j1, l1, i, k] <= self.L[l][j, l, i, k] - self.process_time[j][i] + (3 - self.x[l, l1, k] - self.y[l1][0, j1, l1] - self.y[l][j, 0, l] + self.t[l1][j1, l1, i, k]) * self.max_infinity
                              for l in self.group_array[1:]
                              for l1 in self.group_array[1:]
                              if l != l1
                              for j in self.jobs_in_each_group[l][1:]
                              for j1 in self.jobs_in_each_group[l1][1:]
                              for i in self.machine_array[1:]
                              for k in self.factory_array[1:])

    # 20对应约束22
    def constrants20(self):
        self.model.addConstrs(
            self.L[l][j, l, i, k] >= self.ml[i] - (1 - self.t[l][j, l, i, k]) * self.max_infinity
            for l in self.group_array[1:]
            for j in self.jobs_in_each_group[l][1:]
            for i in self.machine_array[1:]
            for k in self.factory_array[1:])

    # 21对应约束23
    def constrants21(self):
        self.model.addConstrs(
            self.L[l][j, l, i, k] <= self.ml[i] + (1 - self.t[l][j, l, i, k]) * self.max_infinity
            for l in self.group_array[1:]
            for j in self.jobs_in_each_group[l][1:]
            for i in self.machine_array[1:]
            for k in self.factory_array[1:])

    # 22对应约束29
    def constrants22(self):
        self.model.addConstrs(self.c_max >= self.c[l][j, l, self.num_of_machines]
                              for l in self.group_array[1:]
                              for j in self.jobs_in_each_group[l][1:])

    # 23对应约束30
    def constrants23(self):
        self.model.addConstrs(self.mo[k,i] >= self.c[l][j, l ,i] - (1 - self.x[l,0,k]) * self.max_infinity
                              for l in self.group_array[1:]
                              for j in self.jobs_in_each_group[l][1:]
                              for i in self.machine_array[1:]
                              for k in self.factory_array[1:])
    #24对应约束31
    def constrants24(self):
        self.model.addConstrs(
            self.TPE[f] == gp.quicksum(gp.quicksum(gp.quicksum(
                self.process_time[j, i] * self.w[l, f] * gp.quicksum(
                    self.z[l][j, l, i, s] * 4 * self.v1[s] * self.v1[s] * self.v[s] for s in self.speed_array[1:])
                for j in self.jobs_in_each_group[l][1:])
                                                   for l in self.group_array[1:])
                                       for i in self.machine_array[1:])
            for f in self.factory_array[1:])

    # 25对应约束32
    def constrants25(self):
        self.model.addConstrs(
            self.TSE[f] == 0.5 *(gp.quicksum(self.setup_time[i, l, l1] * self.x[l, l1, f]
                        for l in self.group_array[1:] for l1 in self.group_array[1:] if l1 != l for i in self.machine_array[1:]) +
            gp.quicksum(self.setup_time[i, l, l] * self.x[0, l, f]
                        for l in self.group_array[1:] for i in self.machine_array[1:]))
            for f in self.factory_array[1:])

    # 26对应约束33
    def constrants26(self):
        self.model.addConstrs(
            self.TIE[f] == gp.quicksum(self.mo[f, i] for i in self.machine_array[1:]) -
            gp.quicksum(self.w[l,f] * self.process_time[j][i] * gp.quicksum(self.z[l][j, l, i, s] * self.v[s] for s in self.speed_array[1:])
                        for i in self.machine_array[1:]  for l in self.group_array[1:] for j in self.jobs_in_each_group[l][1:]) -
            gp.quicksum(self.setup_time[i, l, l1] * self.x[l, l1, f]
                        for l in self.group_array[1:] for l1 in self.group_array[1:] if l1 != l for i in self.machine_array[1:]) -
            gp.quicksum(self.setup_time[i, l, l] * self.x[0, l, f]
                        for l in self.group_array[1:] for i in self.machine_array[1:])
            - gp.quicksum(self.mt[i] * self.t[l][j, l, i, f] for l in self.group_array[1:] for j in self.jobs_in_each_group[l][1:] for i in self.machine_array[1:])
            for f in self.factory_array[1:])

    # 27对应约束34
    def constrants27(self):
        self.model.addConstr(
            self.TEC >= gp.quicksum(self.TPE[f] + self.TSE[f] + self.TIE[f] for f in self.factory_array[1:]))



    def print_vars(self):
        # print("Model Attributes")
        print("Current optimization status", self.model.Status)
        print("Indicates whether the model is a MIP", self.model.IsMIP)
        print("Indicates whether the model has multiple objectives", self.model.IsMultiObj)
        if self.model.IsMultiObj == 0:
            print("Current relative MIP optimality gap", self.model.MIPGap)
        print("Runtime for most recent optimization", self.model.Runtime)
        print("Work spent on most recent optimization", self.model.Work)

        print("Number of variables", self.model.NumVars)
        print("Number of integer variables", self.model.NumIntVars)
        print("NumBinVars", self.model.NumBinVars)

        print("Number of linear constraints", self.model.NumConstrs)
        print("Number of SOS constraints", self.model.NumSOS)
        print("Number of quadratic constraints", self.model.NumQConstrs)
        print("Number of general constraints", self.model.NumGenConstrs)

        print("Number of non-zero coefficients in the constraint matrix", self.model.NumNZs)
        print("Number of non-zero coefficients in the constraint matrix (in double format)", self.model.DNumNZs)
        print("Number of non-zero quadratic objective terms", self.model.NumQNZs)
        print("Number of non-zero terms in quadratic constraints", self.model.NumQCNZs)

        print("Number of stored solutions", self.model.SolCount)
        print("Number of simplex iterations performed in most recent optimization", self.model.IterCount)
        print("Number of barrier iterations performed in most recent optimization", self.model.NodeCount)
        print("Number of branch-and-cut nodes explored in most recent optimization", self.model.NumQCNZs)
        print("Number of open branch-and-cut nodes at the end of most recent optimization", self.model.OpenNodeCount)

        print("Maximum linear objective coefficient (in absolute value)", self.model.MaxObjCoeff)
        print("MinObjCoeff	Minimum (non-zero) linear objective coefficient (in absolute value)",
              self.model.MinObjCoeff)
        print("Maximum constraint right-hand side (in absolute value)", self.model.MaxRHS)
        print("Minimum (non-zero) constraint right-hand side (in absolute value)", self.model.MinRHS)

        # get the set of variables
        variables = self.model.getVars()

        # Ensure status is optimal
        # assert self.model.Status == GRB.Status.OPTIMAL

        # Query number of multiple objectives, and number of solutions
        nSolutions = self.model.SolCount
        nObjectives = self.model.NumObj
        print('Problem has', nObjectives, 'objectives')
        print('Gurobi found', nSolutions, 'solutions')

        # For each solution, print value of first three variables, and
        # value for each objective function
        solutions = []
        for s in range(nSolutions):
            # Set which solution we will query from now on
            self.model.params.SolutionNumber = s

            # Print objective value of this solution in each objective
            print('Solution', s, ':', end=' ')
            if self.model.IsMultiObj == 0:
                print(self.model.PoolObjVal, end=' ')
            else:
                for o in range(nObjectives):
                    # Set which objective we will query
                    self.model.params.ObjNumber = o
                    # Query the o-th objective value
                    print(self.model.ObjNVal, end=' ')

            # print first ten variables in the solution
            print('->', end=' ')
            n = min(len(variables), 3)
            j = 0
            for v in variables:
                if v.Xn >= 0.9:
                    print(v.VarName, v.Xn, end=' ')
                    j = j + 1
                    if j == n:
                        break
            print('')

            # query the full vector of the o-th solution
            solutions.append(self.model.getAttr('Xn', variables))
        print('Optimal Solution variables')
        if nSolutions > 0:
            for v in variables:
                if v.X >= 0.9:
                    print(v.VarName, v.X)

    def graph(self):
        result = []
        complete_time_of_group = {}
        # 第一部分：初始化机器状态
        for f in self.factory_array[1:]:
            for m in self.machine_array[1:]:
                result.append(
                    {"factory": f,
                     "stage": None,
                     "machine": m,
                     "group": -1,
                     "job": 0,
                     "color_category": 0,
                     "start": 0,
                     "finish": 0,
                     "departure": 0
                     })

        for f in self.factory_array[1:]:
            for m in self.machine_array[1:]:
                for l in self.group_array[1:]:
                    if self.w[l, f].X > 0.9:
                        for l1 in self.group_array:
                            if l1 != l and self.x[l1, l, f].X > 0.9:
                                if l1 == 0:
                                        result.append(
                                            {"factory": f,
                                             "stage": None,
                                             "machine": m,
                                             "group": -1,
                                             "job": -1,
                                             "color_category": -1,
                                             "start": 0,
                                             "finish": self.setup_time[m, l, l],
                                             "departure": self.setup_time[m, l, l], })
                                else:
                                    tmp_j = 0
                                    for j in self.jobs_in_each_group[l1][1:]:
                                        if self.y[l1][j, 0, l1].X > 0.9 :
                                            tmp_j = j
                                            break
                                    result.append(
                                        {"factory": f,
                                         "stage": None,
                                         "machine": m,
                                         "group": -1,
                                         "job": -1,
                                         "color_category": -1,
                                         "start": self.c[l1][tmp_j, l1, m].X ,
                                         "finish": self.c[l1][tmp_j, l1, m].X + self.setup_time[m, l1, l],
                                         "departure": self.c[l1][tmp_j, l1, m].X + self.setup_time[m, l1, l], })

                        # 安排任务处理时间
                            for j in self.jobs_in_each_group[l][1:]:
                                result.append(
                                    {"factory": f,
                                     "stage": None,
                                     "machine": m,
                                     "group": l,
                                     "job": j,
                                     "color_category": l,
                                     "start": self.c[l][j, l, m].X - self.process_time[j, m] * sum(
                                         self.z[l][j, l, m, s].X * self.v[s] for s in self.speed_array[1:]),
                                     "finish": self.c[l][j, l, m].X,
                                     "departure": self.c[l][j, l, m].X
                                     })

                        # 在任务处理后安排维护任务
                            for j in self.jobs_in_each_group[l][1:]:
                                if self.t[l][j, l, m, f].X > 0.9:  # 判断是否需要安排维护
                                    # start_maintenance = self.c[l][j, l, m].X + self.setup_time[m, l, l1]
                                    result.append(
                                        {"factory": f,
                                         "stage": None,
                                         "machine": m,
                                         "group": -1,
                                         "job": -2,  # 表示维护任务
                                         "color_category": -1,
                                         "start": self.c[l][j, l, m].X - self.process_time[j, m] * sum(
                                         self.z[l][j, l, m, s].X * self.v[s] for s in self.speed_array[1:]) - self.mt[m],
                                         "finish": self.c[l][j, l, m].X - self.process_time[j, m] * sum(
                                         self.z[l][j, l, m, s].X * self.v[s] for s in self.speed_array[1:]),
                                         "departure": self.c[l][j, l, m].X - self.process_time[j, m] * sum(
                                         self.z[l][j, l, m, s].X * self.v[s] for s in self.speed_array[1:]),
                                         })

        # 生成 Gantt 图
        df = pd.DataFrame(result)
        max_finish = df.finish.max()

        set_gantt_color(df, palette="pastel6")

        fig, axes = plt.subplots(self.num_of_factories, 1, figsize=(20, 5))
        for f in self.factory_array[1:]:
            f_df = df[df.factory == f]
            plt.sca(axes[f - 1])
            gantt(data=f_df, show_title=True,
                  max_finish=max_finish)
        plt.tight_layout()
        plt.show()




if __name__ == '__main__':
    file = FGSPModel('Se_FG_2_20_4.txt')
    file.creat_model()


