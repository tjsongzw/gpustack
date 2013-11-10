"""
A layer that can be _pretrained_ as an RBM.
"""


from gnumpy import dot as gdot
from gnumpy import zeros as gzeros
from gnumpy import sum as gsum
import gnumpy as gpu
import numpy as np


from rbm import RBM
from misc import match_table, bernoulli
from misc import diff_table, cpu_table, str_table


class SPATIAL_RBM(RBM):
    def __init__(self, shape, activ=None, params=None, **kwargs):
        super(SPATIAL_RBM, self).__init__(shape=shape, activ=activ, params=params, **kwargs)

    def __repr__(self):
        hrep = str(self.H).split()[1]
        vrep = str(self.V).split()[1]
        rep = "Spatial-RBM-%s-%s-%s-[sparsity--%s:%s]-[block--%s-%s]"%(vrep, hrep, self.shape, self.lmbd, self.rho, self.block, self.block_shape)
        if hasattr(self, 'drawin'):
            rep += "-[drawin--%s]"%(self.drawin)
        return rep


    def pt_init(self, block, drawin=None, H=bernoulli, V=bernoulli,
                init_var=1e-2, init_bias=0., rho=0.5, lmbd=0.,
                l2=0., **kwargs):
        pt_params = gzeros(self.m_end + self.shape[1] + self.shape[0])
        if init_var is None:
            init_heur = 4*np.sqrt(6./(self.shape[0]+self.shape[1]))
            pt_params[:self.m_end] = gpu.rand(self.m_end)
            pt_params[:self.m_end] *= 2
            pt_params[:self.m_end] -= 1
            pt_params[:self.m_end] *= init_heur
        else:
            pt_params[:self.m_end] = init_var * gpu.randn(self.m_end)
        pt_params[self.m_end:] = init_bias

        self.H = H
        self.V = V
        self.activ = match_table[H]

        self.pt_score = self.reconstruction
        self.pt_grad = self.grad_cd1

        self.l2 = l2

        self.rho = rho
        self.lmbd = lmbd
        self.rho_hat = None
        self.block = int(block)
        self.block_shape = (self.shape[0]/self.block, self.shape[1]/self.block)
        _mask = gzeros(self.shape)
        for i in xrange(self.block):
            _mask[i*self.block_shape[0]:(i+1)*self.block_shape[0],
                 i*self.block_shape[1]:(i+1)*self.block_shape[1]] = 1
        self.mask = _mask.ravel()
        self.mask_stable = _mask.ravel().copy()
        if drawin is not None and drawin > 0:
            assert(0 < drawin < 1), "drawin needs to be in (0,1)."
            self.drawin = drawin
        if drawin is None:
            pt_params[:self.m_end] *= self.mask_stable

        return pt_params


    def grad_cd1(self, params, inputs, **kwargs):
        """
        """
        if hasattr(self, 'drawin'):
            self.mask = self.mask_stable.copy()
            draw = (gpu.rand(self.shape) < self.drawin).ravel()
            inverse_mask = -1. * (self.mask_stable - 1)
            draw *= inverse_mask
            self.mask += draw

        g = gzeros(params.shape)

        n, _ = inputs.shape

        m_end = self.m_end
        V = self.shape[0]
        H = self.shape[1]
        # wm = params[:m_end].reshape(self.shape)
        wm = (params[:m_end]*self.mask).reshape(self.shape)

        h1, h_sampled = self.H(inputs, wm=wm, bias=params[m_end:-V], sampling=True)
        v2, _ = self.V(h_sampled, wm=wm.T, bias=params[-V:])
        h2, _ = self.H(v2, wm=wm, bias=params[m_end:-V])

        # Note the negative sign: the gradient is
        # supposed to point into 'wrong' direction,
        # because the used optimizer likes to minimize.
        g[:m_end] = -gdot(inputs.T, h1).ravel()
        g[:m_end] += gdot(v2.T, h2).ravel()
        g[:m_end] *= 1./n
        g[:m_end] += self.l2*params[:m_end]


        g[m_end:-V] = -h1.mean(axis=0)
        g[m_end:-V] += h2.mean(axis=0)

        g[-V:] = -inputs.mean(axis=0)
        g[-V:] += v2.mean(axis=0)

        if self.rho_hat is None:
            self.rho_hat = h1.mean(axis=0)
        else:
            self.rho_hat *= 0.9
            self.rho_hat += 0.1 * h1.mean(axis=0)
        dKL_drho_hat = (self.rho - self.rho_hat)/(self.rho_hat * (1 - self.rho_hat) + 1e-10)
        h1_1mh1 = h1*(1 - h1)
        g[m_end:-V] -= self.lmbd/n * gsum(h1_1mh1, axis=0) * dKL_drho_hat
        g[:m_end] -= self.lmbd/n * (gdot(inputs.T, h1_1mh1) * dKL_drho_hat).ravel()


        # #spatial rbm parameter
        # g[:m_end] *= self.mask
        wm_block = gzeros(self.block_shape)

        wm_unravel = g[:m_end].reshape(self.shape)
        h_block = g[m_end:-V].reshape((self.block, -1)).mean(axis=0)
        v_block = g[-V:].reshape((self.block, -1)).mean(axis=0)

        for i in xrange(self.block):
            wm_block += wm_unravel[i*self.block_shape[0]:(i+1)*self.block_shape[0],
                                  i*self.block_shape[1]:(i+1)*self.block_shape[1]]

        wm_block *= 1./self.block

        wm_unravel *= self.mask.reshape(self.shape)

        for i in xrange(self.block):
            wm_unravel[i*self.block_shape[0]:(i+1)*self.block_shape[0],
                       i*self.block_shape[1]:(i+1)*self.block_shape[1]] = wm_block

        g[:m_end] = wm_unravel.ravel()
        g[m_end:-V] = gpu.tile(h_block, self.block)
        g[-V:] = gpu.tile(v_block, self.block)

        return g


    def fprop(self, params, data):
        if hasattr(self, 'drawin'):
            self.mask = self.mask_stable.copy()
            draw = (gpu.rand(self.shape) < self.drawin).ravel()
            inverse_mask = -1. * (self.mask_stable - 1)
            draw *= inverse_mask
            self.mask += draw

        self.data = data
        self.Z = self.activ(gdot(data, (params[:self.m_end]*self.mask).reshape(self.shape)) + params[self.m_end:])
        return self.Z


    def bprop(self, params, grad, delta):
        dE_da = self.class_weight * delta * diff_table[self.activ](self.Z)
        # gradient of the bias
        grad[self.m_end:] = dE_da.sum(axis=0)
        # gradient of the weights
        grad[:self.m_end] = gdot(self.data.T, dE_da).ravel()
        # spatial rbm
        grad[:self.m_end] *= self.mask
        # backpropagate the delta
        delta = gdot(dE_da, params[:self.m_end].reshape(self.shape).T)
        del self.Z
        return delta

    def bprop_dropout(self, params, grad, delta):
        delta *= self.drop
        dE_da = self.class_weight * delta * diff_table[self.activ](self.Z)
        # gradient of the bias
        grad[self.m_end:] = dE_da.sum(axis=0)
        # gradient of the weights
        grad[:self.m_end] = gdot(self.data.T, dE_da).ravel()

        grad[:self.m_end] *= self.mask
        # backpropagate the delta
        delta = gdot(dE_da, params[:self.m_end].reshape(self.shape).T)
        del self.Z
        del self.drop
        return delta
