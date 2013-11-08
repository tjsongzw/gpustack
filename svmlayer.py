"""
See: http://www.cs.toronto.edu/~tang/papers/dlsvm.pdf
Deep Learning using Support Vector Machines, Charlie Tang.
"""


import numpy as np


from gnumpy import dot as gdot
from gnumpy import zeros as gzeros
import gnumpy as gpu


from misc import diff_table, cpu_table, str_table, idnty
from layer import Layer


class SVMLayer(Layer):
    def __init__(self, shape, C, params=None, dropout=None, class_weight=None, **kwargs):
        super(SVMLayer, self).__init__(shape=shape, activ=idnty, params=params,
                                       dropout=dropout, class_weight=class_weight)
        self.C = C

    def __repr__(self):
        if self.score is None:
            _score = "no_score"
        else:
            _score = str(self.score).split()[1]
        return "SVMLayer-%s-%s-%s"%(_score, str_table[self.activ], self.shape)

    def fward(self, params, data):
        return gdot(data, params[:self.m_end].reshape(self.shape)) + params[self.m_end:]

    def fprop(self, params, data):
        self.data = data
        self.Z = gdot(data, params[:self.m_end].reshape(self.shape)) + params[self.m_end:]
        return self.Z

    def bprop(self, params, grad, delta):
        # TODO: check next line, is it according
        # to formula in the paper? delta must be
        # defined correctly!!
        # self.C necessary? in loss Function, there is no C
        dE_da = self.C * delta * diff_table[self.activ](self.Z)
        # gradient of the bias
        grad[self.m_end:] = dE_da.sum(axis=0)
        # gradient of the weights, takes care of weight 'decay' factor (second addend)
        grad[:self.m_end] = gdot(self.data.T, dE_da).ravel() + params[:self.m_end]
        # backpropagate the delta
        delta = gdot(dE_da, params[:self.m_end].reshape(self.shape).T)
        del self.Z
        return delta

    def pt_score(self, params, inputs, targets, l2=0, **kwargs):
        Z = self.activ(gpu.dot(inputs, params[:self.m_end].reshape(self.shape)) + params[self.m_end:])
        sc = self.score(Z, targets)
        # necessary? in loss Function, there is no C
        sc = self.C * sc
        return sc

    def pt_grad(self, params, inputs, targets, l2=0, **kwargs):
        g = gzeros(params.shape)
        Z = self.activ(gpu.dot(inputs, params[:self.m_end].reshape(self.shape)) + params[self.m_end:])
        _, delta = self.score(Z, targets, error=True)
        # necessary?
        delta = self.C * delta
        g[:self.m_end] = gdot(inputs.T, delta).ravel() + params[:self.m_end]
        g[self.m_end:] = delta.sum(axis=0)
        # clean up
        del delta
        return g
