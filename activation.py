"""
    activation function for machine learning
"""
from abc import ABCMeta, abstractmethod
import numpy as np

class Base(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x):
        pass
    @abstractmethod
    def backward(self, x):
        pass
    
    def auto_forward(self, x):
        self.y = self.forward(x)
        return self.y
    
    def auto_backward(self):
        return self.backward(self.y)

class Sigmoid(Base):
    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))
    @staticmethod
    def backward(x):
        return x * (1 - x)
