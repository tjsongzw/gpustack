from gnumpy import dot as gdot
from gnumpy import zeros as gzeros
from gnumpy import sum as gsum
import gnumpy as gpu
import numpy as np


from rbm import RBM
from misc import match_table, bernoulli, multinomial


class RSM(RBM):
    def __init__(self, shape, activ=None, params=None, **kwargs):
        super(RSM, self).__init__(shape=shape, activ=activ,
                                  params=params, **kwargs)

    def __repr__(self):
        hrep = str(self.H).split()[1]
        vrep = str(self.V).split()[1]
        rep = ("RSM-%s-%s-%s-[sparsity--%s:%s]" %
               (vrep, hrep, self.shape, self.lmbd, self.rho))
        return rep

    def pt_init(self, H=bernoulli, V=multinomial, init_var=1e-5,
                init_bias=0., rho=0.5, lmbd=0., l2=0., **kwargs):
        return super(RSM, self).pt_init(H=H, V=V, init_var=init_var,
                                        init_bias=init_bias, rho=rho,
                                        lmbd=lmbd, l2=l2, **kwargs)

    def reconstruction(self, params, inputs, **kwargs):
        """
        """
        D = inputs.sum(axis=1)
        bias = gpu.outer(D, params[self.m_end:-self.shape[0]])
        h1, h_sampled = self.H(inputs,
                               wm=params[:self.m_end].reshape(self.shape),
                               bias=bias, sampling=True)
        _, v2 = self.V(h_sampled,
                       wm=params[:self.m_end].reshape(self.shape).T,
                       bias=params[-self.shape[0]:], n=D, sampling=True)

        rho_hat = h1.mean()
        rec = ((inputs - v2)**2).sum()

        return np.array([rec, rho_hat])

    def grad_cd1(self, params, inputs, **kwargs):
        """
        """
        g = gzeros(params.shape)

        n, _ = inputs.shape

        m_end = self.m_end
        V = self.shape[0]
        H = self.shape[1]
        wm = params[:m_end].reshape(self.shape)
        D = inputs.sum(axis=1)
        bias = gpu.outer(D, params[m_end:-V])
        h1, h_sampled = self.H(inputs, wm=wm, bias=bias, sampling=True)
        _, v2 = self.V(h_sampled, wm=wm.T,
                       bias=params[-V:], n=D, sampling=True)
        h2, _ = self.H(v2, wm=wm, bias=bias)

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
        dKL_drho_hat = (self.rho - self.rho_hat)/(self.rho_hat*(1-self.rho_hat))
        h1_1mh1 = h1*(1 - h1)
        g[m_end:-V] -= self.lmbd/n * gsum(h1_1mh1, axis=0) * dKL_drho_hat
        g[:m_end] -= self.lmbd/n * (gdot(inputs.T, h1_1mh1) * dKL_drho_hat).ravel()

        return g
