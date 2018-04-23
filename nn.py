from activation import Sigmoid
import numpy as np

class BP:
    def __init__(self, learning_rate=0.01, iteration=1000, log_step=None):
        self.weights = []                   # 三维数组，仅含隐含层、输出层。1: 层数; 2: 当前层神经元序号; 3: 上一层神经元序号
        self.acfuncs = []                   # 一维数组
        self.learning_rate = learning_rate  # 训练速度
        self.iteration = iteration          # 迭代次数
        self.log_step = log_step

    def add_layer(self, in_num, out_num, init_weight=0.01, init_bias=0.01, activation_func=None):

        # 初始化神经网络模型，即weight、bias组成的二维数组
        layer = np.array([[init_weight for _ in range(in_num+1)] for _ in range(out_num)])
        for cur in layer:
            cur[-1] = init_bias

        self.weights.append(layer)
        self.acfuncs.append(activation_func)
    
    def train(self, x, y):

        # 创建二维数组，记录每层每个神经元的输出，重写于正向传播，使用于反向传播
        n_layers = len(self.weights)
        vals = []                           # 记录每层每个神经元的输出，包括输入层的x
        vals.append(np.copy(x))             # 输入层的输出
        for i in range(n_layers):
            n_units = len(self.weights[i])  # 该层神经元数
            vals.append(np.array([0.0 for _ in range(n_units)]))

        for it in range(1, 1+self.iteration):

            self.forward(x, vals)           # 正向传播
            self.backward(y, vals)          # 反向传播

            if self.log_step!=None and (it%self.log_step==0 or it==1 or it == self.iteration):
                print("progress: %d %%, cost: %.4f" % (100*it/self.iteration, self.get_cost(x, y, vals)), flush=True)

    # 正向传播，得到预测序列
    def forward(self, x, vals):
        in_val = np.copy(x)
        for i, layer in enumerate(self.weights):                        # 遍历神经网络的每一层
            in_val = np.append(in_val, 1.0)

            for j, wbs in enumerate(layer): # 遍历每层的每个神经元，与每个神经元相关的是分别在前后连接的权重
                out = np.dot(in_val, wbs)   # 临时变量，每个神经元的输出，下一层的输入。矩阵点乘法则恰好符合运算需求
                if self.acfuncs[i] != None:   # 激活函数
                    out = self.acfuncs[i].forward(out)
                vals[i+1][j] = out
            
            in_val = vals[i+1] # +1原因: model数组不包括输入层，out数组包括输入层

    # 计算损失，通过正向传播
    def get_cost(self, x, y, out):
        self.forward(x, out)
        cost = sum((out[-1] - y)**2)
        return cost

    # 反向传播，更新权重
    def backward(self, y, vals):

        funcs = self.acfuncs
        lr = self.learning_rate
        ws = self.weights
        n_layers = len(ws)

        for i in range(n_layers-1, -1, -1):   # 倒序遍历所有神经元层，“反向”为此意

            out_val = vals[i+1] # 该层的输出
            in_val = vals[i]    # 该层的输入
            in_val = np.append(in_val, 1.0)

            if i == n_layers-1:
                e = y - vals[-1]
            else:
                e = [sum(wbs * g[j]) for j, wbs in enumerate(ws[i])]
            
            g = funcs[i].backward(out_val) * e
            for j, wbs in enumerate(ws[i]):
                wbs += lr * (g[j] * in_val)

if __name__ == "__main__":
    x = np.linspace(1, 10, 30)
    y = (np.sin(x) + 1) / 2
    
    # 训练
    nn = BP(learning_rate=0.01, iteration=1000, log_step=100)
    nn.add_layer(30, 30, activation_func=Sigmoid)
    nn.add_layer(30, 30, activation_func=Sigmoid)
    nn.train(x, y)
