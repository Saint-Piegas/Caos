import numpy as np
from numba import int32, float64
from numba.experimental import jitclass
import matplotlib.pyplot as plt
import time

spec = [
    ('N', int32),
    ('NT', int32),
    ('a', float64[:]),
    ('eps', float64),
    ('FATOR', int32)
]
@jitclass(spec)
class Mapa:
    def __init__(self, N, NT, a, eps = 0, FATOR = 5):
        self.N = N
        self.NT = NT
        self.a = a
        self.eps = eps
        self.FATOR = FATOR
    
    def f(self, x, a):
        return a*x*(1-x)
    
    def dfdx(self, x, a):
        return a*(1-2*x)
    
    def run(self, x0):
        tamanho_a = len(self.a)
        x = np.zeros((self.N+self.NT+1, tamanho_a))
        L = np.zeros(tamanho_a)
        x[0,:] = x0
        for i in range(self.N+self.NT):
            x[i+1, :] = self.f(x[i], self.a)
            if i - self.NT >= 0:
                L += 1/tamanho_a * np.log(np.abs(self.dfdx(x[i], self.a)) + 1e-15)
        x = x[self.NT+1:, :]
        
        if self.eps == 0:
            return x.T, L
        
        novo_NT = self.NT * self.FATOR
        for indice in range(len(L)):
            #Confere se o coeficiente é menor que o epsilon para melhorar a precisão (toma mais valores de )
            if np.abs(L[indice]) <= self.eps:
                x[0, indice] = x[-1, indice]
                for i in range(self.N + novo_NT - 1):
                    if i - novo_NT >= 0:
                        x[i - novo_NT +1, indice] = self.f(x[i - novo_NT, indice], self.a[indice])
                        L[indice] += 1/tamanho_a * np.log(np.abs(self.dfdx(x[i - novo_NT, indice], self.a[indice])) + 1e-15)
        return x.T, L
    
    def cobweb(self, its):
        pass
    
class DiagramaDeBifurcacao:
    def __init__():
        pass

As = np.linspace(2.8, 3.20, 1_000)[1:]
mapa = Mapa(N = 5000, NT = 1_000_000, a = As, eps = 0.5, FATOR=50)
x, L = mapa.run(np.random.rand(len(As)))

plt.plot(np.repeat(mapa.a, x.shape[1]), x.flatten(), ',', color='k', alpha=0.02, markersize=1 )
plt.show()
