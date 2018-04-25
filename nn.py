from activation import Sigmoid
import numpy as np
import pickle
import os

class BP:
    def __init__(self, in_num=None):
        """
            (BP, error Back Propagation) The basic realization of BP neural network.

            in_num      : number of neurons in the input layer.
        """
        self.weights = []           # 三维数组，仅含隐含层、输出层。1: 层数; 2: 当前层神经元序号; 3: 上一层神经元序号
        self.acfuncs = []           # 一维数组，每层的激活函数
        self.__in_num = in_num      # 输入层神经元数。输出层神经元数由最后调用的add_layer决定。
                                    # self.__in_num在每次add_layer时都会改变

    # 新增神经元层
    def add_layer(self, n_units, init_weight=None, init_bias=None, ac_func=None):
        """
            Adding neuronal layer to neural network.

            n_units     : number of neurons in the new layer
            init_weight : weight initial value
            init_bias   : bias initial value
            ac_func     : activation function
        """

        # 初始化神经网络模型，即weight、bias组成的二维数组
        weights = np.array([self.__init_wb(self.__in_num, init_weight) for _ in range(n_units)])
        biases = self.__init_wb(n_units, init_bias, transpose=True)
        layer = np.hstack((weights, biases))

        self.weights.append(layer)
        self.acfuncs.append(ac_func)

        self.__in_num = n_units
    
    # 训练
    def train(self, xs, ys, learning_rate=0.01, iteration=1000, log_step=None):
        """
            Start training model. Sequential mode.

            learning_rate   : learning rate
            iteration       : total number of iterations
            log_step        : interval of logging loss. If log_step is None, it will not log
        """

        # 创建二维数组，记录每层每个神经元的输出，重写于正向传播，使用于反向传播。避免频繁申请内存
        vals = self.__create_vals(xs[0])

        for it in range(1, 1+iteration):
            for t, _ in enumerate(xs):
                self.__forward(xs[t], vals)                 # 正向传播
                self.__backward(ys[t], learning_rate, vals) # 反向传播

            if log_step!=None and (it%log_step==0 or it==1 or it == iteration):
                print("progress: %6.2f %%, cost: %.4f" % (100*it/iteration, self.get_cost(xs, ys, vals)), flush=True)
    
    # 预测
    def predict(self, x):
        """
            Predict result.
        """
        vals = self.__create_vals(x)
        self.__forward(x, vals)
        return tuple(vals[-1])
    
    # 保存模型
    def save(self, filepath):
        """
            Saving the model to a file.
        """
        
        filepath = os.path.abspath(filepath)
        data = {}
        data["weight"] = self.weights
        data["acfunc"] = self.acfuncs
        data["in_num"] = self.__in_num
        pickle.dump(data, open(filepath, "wb"))

        return filepath
    
    # 加载模型
    def load(self, filepath):
        """
            Loading the model from a file
        """

        data = pickle.load(open(filepath, "rb"))
        self.weights = data["weight"]
        self.acfuncs = data["acfunc"]
        self.__in_num = data["in_num"]

        return True
    
    # 初始化权重
    def __init_wb(self, num, init_val=None, transpose=False):
        """
            init weights or biases, return np.array.
        """
        if init_val is None:
            wb = np.random.normal(loc=0.0, scale=1.0, size=num)
        else:
            wb = np.array([init_val for _ in range(num)])
        
        if transpose:
            wb = wb.reshape((num, 1))
        
        return wb

    # 创建二维数组，用于存储中间结果
    def __create_vals(self, x):
        """
            Create 2d array saving intermediate results.
        """
        n_layers = len(self.weights)
        vals = []                           # 记录每层每个神经元的输出，包括输入层的x
        vals.append(np.copy(x))             # 输入层的输出
        for i in range(n_layers):
            n_units = len(self.weights[i])  # 该层神经元数
            vals.append(np.array([0.0 for _ in range(n_units)]))
        return vals

    # 计算损失，通过正向传播
    def get_cost(self, xs, ys, vals):
        """
            Calculate the loss.
        """
        cost = 0
        for t, x in enumerate(xs):
            self.__forward(x, vals)
            cost += np.sum((vals[-1] - ys[t])**2)
        return cost/len(xs)

    # 正向传播，得到预测序列
    def __forward(self, x, vals):
        """
            Positive propagation.
        """
        
        in_val = np.copy(x)
        for j, _ in enumerate(x):
            vals[0][j] = x[j]

        for i, layer in enumerate(self.weights):    # 遍历神经网络的每一层
            in_val = np.append(in_val, 1.0)

            for j, wbs in enumerate(layer):         # 遍历每层的每个神经元，与每个神经元相关的是分别在前后连接的权重
                out = np.dot(in_val, wbs)           # 临时变量，每个神经元的输出，下一层的输入。矩阵点乘法则恰好符合运算需求
                if self.acfuncs[i] != None:         # 激活函数
                    out = self.acfuncs[i].forward(out)
                vals[i+1][j] = out
            
            in_val = vals[i+1] # +1原因: model数组不包括输入层，out数组包括输入层

    # 反向传播，更新权重
    def __backward(self, y, lr, vals):
        """
            Error back propagation.
        """

        funcs = self.acfuncs
        ws = self.weights
        n_layers = len(ws)

        for i in range(n_layers-1, -1, -1):   # 倒序遍历所有神经元层，“反向”为此意

            out_val = vals[i+1] # 该层的输出
            in_val = vals[i]    # 该层的输入
            in_val = np.append(in_val, 1.0)

            if i == n_layers-1: # 输出层
                e = y - vals[-1]
            else:               # 除输出层之外的层
                e = np.array([np.sum(np.multiply(g, ws[i+1][:,j])) for j, _ in enumerate(ws[i])])
            
            if funcs[i] is None:
                g = e
            else:
                g = np.multiply(funcs[i].backward(out_val), e)
            
            for j, wbs in enumerate(ws[i]):
                wbs += lr * (np.multiply(g[j], in_val))
