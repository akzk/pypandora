"""
    1. linear regression
    2. logistic regression
"""

from abc import ABCMeta, abstractmethod
from activation import Sigmoid
import numpy as np
import pickle

class Base(metaclass=ABCMeta):
    """
        Base Linear Model
    """
    def __init__(self):
        self.ws = None
    
    @abstractmethod
    def get_ys(self, xs):
        pass
    
    @abstractmethod
    def get_cost(self, xs, ys):
        pass
    
    def extand_one(self, xs):
        """
            对每个样本的x序列加上1，起到求y公式中bias的作用
        """
        return np.hstack((xs, [[1.0] for _ in enumerate(xs)]))
    
    def train(self, xs, ys, init_val=0.1, learning_rate=0.0001, iteration=100000, log_step=None):
        """
            train LinearRegression model
        """
        xs = self.extand_one(xs)

        # 权重(w、b都有)数等于每个样本的特征数
        self.ws = np.array([init_val for _ in enumerate(xs[0])])

        for iter in range(1, 1+iteration):

            ys_ = self.get_ys(xs)
            loss = ys_ - ys

            for i, _ in enumerate(self.ws):
                self.ws[i] -= learning_rate * np.sum(loss * xs[:,i]) / len(xs)
            
            if log_step != None and (iter % log_step == 0 or iter == 1 or iter == iteration):
                print("progress: %.2f%% lost: %.2f" % (100*iter/iteration, self.get_cost(xs, ys)), flush=True)
    
    def predict(self, x):
        ys_ = self.get_ys(self.extand_one([x]))
        return ys_[0]
    
    def save(self, filepath):
        """
            save LR model
        """
        pickle.dump(self.ws, open(filepath, 'wb'))

    def load(self, filepath):
        """
            load LR model
        """
        self.ws = pickle.load(open(filepath, 'rb'))

class LinearRegression(Base):
    def get_ys(self, xs):
        """
            formula: y_m = w_m1*x_m1 + w_m2*x_m2 + ... + w_md*x_md + bias
            input format: [[x_11, x_12, ..., x_1d], ...,[x_m1, x_m2, ..., x_md]]
        """
        return [np.dot(self.ws, xs[i]) for i, _ in enumerate(xs)]
    
    def get_cost(self, xs, ys):
        return sum(([np.dot(self.ws, xs[i]) for i, _ in enumerate(xs)]-ys)**2)/len(xs)

class LogisticRegression(Base):
    def __init__(self):
        self.ws = None
        self.sigmoid = Sigmoid()

    def get_ys(self, xs):
        """
            formula: y_m = sigmoid(w_m1*x_m1 + w_m2*x_m2 + ... + w_md*x_md + bias)
            input format: [[x_11, x_12, ..., x_1d], ...,[x_m1, x_m2, ..., x_md]]
        """
        return self.sigmoid.forward(np.array([np.dot(self.ws, xs[i]) for i, _ in enumerate(xs)]))

    def get_cost(self, xs, ys):
        ys_ = self.get_ys(xs)
        return sum(-ys * np.log(ys_) - (1-ys) * np.log(1-ys_))/len(xs)
