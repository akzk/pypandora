from activation import Sigmoid
import numpy as np

class BP:
    def __init__(self, in_num, learning_rate=0.01, iteration=1000, log_step=None):
        """
            (BP, error Back Propagation) The basic realization of BP neural network.

            in_num          : number of neurons in the inpu layer
            learning_rate   : learning rate
            iteration       : total number of iterations
            log_step        : interval of logging loss. If log_step is None, it will not log
        """
        self.weights = []                   # 三维数组，仅含隐含层、输出层。1: 层数; 2: 当前层神经元序号; 3: 上一层神经元序号
        self.acfuncs = []                   # 一维数组，每层的激活函数
        self.learning_rate = learning_rate  # 学习率
        self.iteration = iteration          # 迭代总次数
        self.log_step = log_step            # 打印损失的迭代间隔
        self.__in_num = in_num              # 输入层神经元数。输出层神经元数由最后调用的add_layer决定。
                                            # self.__in_num在每次add_layer时都会改变

    def add_layer(self, n_units, init_weight=0.01, init_bias=0.01, ac_func=None):
        """
            Adding neuronal layer to neural network.

            n_units     : number of neurons in the new layer
            init_weight : weight initial value
            init_bias   : bias initial value
            ac_func     : activation function
        """

        # 初始化神经网络模型，即weight、bias组成的二维数组
        layer = np.array([[init_weight for _ in range(self.__in_num+1)] for _ in range(n_units)])
        for cur in layer:
            cur[-1] = init_bias

        self.weights.append(layer)
        self.acfuncs.append(ac_func)

        self.__in_num = n_units
    
    def train(self, x, y):
        """
            Start training model.
        """

        # 创建二维数组，记录每层每个神经元的输出，重写于正向传播，使用于反向传播
        n_layers = len(self.weights)
        vals = []                           # 记录每层每个神经元的输出，包括输入层的x
        vals.append(np.copy(x))             # 输入层的输出
        for i in range(n_layers):
            n_units = len(self.weights[i])  # 该层神经元数
            vals.append(np.array([0.0 for _ in range(n_units)]))

        for it in range(1, 1+self.iteration):

            self.__forward(x, vals)           # 正向传播
            self.__backward(y, vals)          # 反向传播

            if self.log_step!=None and (it%self.log_step==0 or it==1 or it == self.iteration):
                print("progress: %6.2f %%, cost: %.4f" % (100*it/self.iteration, self.get_cost(x, y, vals)), flush=True)

    # 计算损失，通过正向传播
    def get_cost(self, x, y, out):
        self.__forward(x, out)
        cost = sum((out[-1] - y)**2)
        return cost

    # 正向传播，得到预测序列
    def __forward(self, x, vals):
        """
            Positive propagation.
        """
        
        in_val = np.copy(x)
        for i, layer in enumerate(self.weights):    # 遍历神经网络的每一层
            in_val = np.append(in_val, 1.0)

            for j, wbs in enumerate(layer):         # 遍历每层的每个神经元，与每个神经元相关的是分别在前后连接的权重
                out = np.dot(in_val, wbs)           # 临时变量，每个神经元的输出，下一层的输入。矩阵点乘法则恰好符合运算需求
                if self.acfuncs[i] != None:         # 激活函数
                    out = self.acfuncs[i].forward(out)
                vals[i+1][j] = out
            
            in_val = vals[i+1] # +1原因: model数组不包括输入层，out数组包括输入层

    # 反向传播，更新权重
    def __backward(self, y, vals):
        """
            Error back propagation
        """

        funcs = self.acfuncs
        lr = self.learning_rate
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
            
            g = np.multiply(funcs[i].backward(out_val), e)
            for j, wbs in enumerate(ws[i]):
                wbs += lr * (np.multiply(g[j], in_val))

if __name__ == "__main__":
    x = np.linspace(1, 10, 30)
    y = (np.sin(x) + 1) / 2
    
    # 训练
    nn = BP(30, learning_rate=0.01, iteration=1000, log_step=64)
    nn.add_layer(64, ac_func=Sigmoid)
    nn.add_layer(30, ac_func=Sigmoid)
    nn.train(x, y)
