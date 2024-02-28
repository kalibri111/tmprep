from nn.module.parameters import Parameters
import numpy as np

class BatchNorm:
    """Реализует Batch norm

    ---------
    Параметры
    ---------
    in_dim : int
        Размерность входного вектора

    eps : float (default=1e-5)
        Параметр модели,
        позволяет избежать деления на 0

    momentum : float (default=0.1)
        Параметр модели
        Используется для обновления статистик
    """

    def __init__(self, in_dim, eps=1e-5, momentum=0.1):
        self.in_dim = in_dim
        self.eps = eps
        self.momentum = 0.1

        self.regime = "Train"

        self.gamma = Parameters((in_dim,))
        self.gamma._init_params()

        self.beta = Parameters(in_dim)

        self.E = np.zeros(in_dim)
        self.D = np.zeros(in_dim)

        self.inpt_hat = None
        self.tmp_D = None
        self.tmp_E = None

    def forward(self, inpt):
        """Реализует forward-pass

        ---------
        Параметры
        ---------
        inpt : np.ndarray, shape=(M, N_in)
            Входные данные

        ----------
        Возвращает
        ----------
        output : np.ndarray, shape=(M, N_in)
            Выход слоя
        """
        if self.regime == "Eval":
            self.inpt_hat = (inpt - self.E) / np.sqrt(self.D + self.eps)
            out = self.inpt_hat * self.gamma.params + self.beta.params
            return out

        self.tmp_E = np.mean(inpt, axis=0)
        self.tmp_D = np.var(inpt, axis=0)
        self.inpt_hat = (inpt - self.tmp_E) / np.sqrt(self.tmp_D + self.eps)
        out = self.inpt_hat * self.gamma.params + self.beta.params
        self.E = (1 - self.momentum) * self.E + self.momentum * self.tmp_E
        self.D = (1 - self.momentum) * self.D + self.momentum * self.tmp_D

        return out

    def __call__(self, *inpt):
        """Аналогично forward"""
        return self.forward(*inpt)

    def parameters(self):
        """Возвращает параметры модели"""
        return (self.gamma, self.beta)

    def _zero_grad(self):
        """Обнуляет градиенты модели"""
        self.gamma.grads = np.zeros(self.gamma.shape)
        self.beta.grads = np.zeros(self.beta.shape)

    def _compute_gradients(self, grads):
        """Считает градиенты модели"""
        if self.regime == "Eval":
            raise RuntimeError("Нельзя посчитать градиенты в режиме оценки")
        
        batch_size = self.inpt_hat.shape[0]
        xmu = self.inpt_hat * np.sqrt(self.tmp_D + self.eps)
        ivar = 1. / np.sqrt(self.tmp_D + self.eps)

        #step9
        self.beta.grads = np.sum(grads, axis=0)
        dgammax = grads 
        #step8
        self.gamma.grads = np.sum(dgammax * self.inpt_hat, axis=0)
        dxhat = dgammax * self.gamma.params
        #step7
        divar = np.sum(dxhat * xmu, axis=0)
        dxmu1 = dxhat * ivar
        #step6
        dsqrtvar = -1. / (self.tmp_D + self.eps) * divar
        #step5
        dvar = 0.5 * ivar * dsqrtvar
        #step4
        dsq = 1. / batch_size * np.ones((batch_size, self.in_dim)) * dvar
        #step3
        dxmu2 = 2 * xmu * dsq
        #step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dx1, axis=0)
        #step1
        dx2 = 1. / batch_size * np.ones((batch_size, self.in_dim)) * dmu
        #step0
        input_grads = dx1 + dx2

        return input_grads

    def _train(self):
        """Переводит модель в режим обучения"""
        self.regime = "Train"

    def _eval(self):
        """Переводит модель в режим оценивания"""
        self.regime = "Eval"

    def __repr__(self):
        return f"BatchNorm(in_dim={self.in_dim}, eps={self.eps})"