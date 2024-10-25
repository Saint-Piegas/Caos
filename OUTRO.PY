import numpy as np
#@title calcula Mapa 1D (calculaMapa1D)

#@title Contador de períodos (contadorPeriodo)
def contadorPeriodo( x, nPMax=np.inf, tolerancia=1e-7, naoPeriodico=0):
  x = np.array(x)[::-1]
  nPMaxEf = int( min( nPMax, len(x)/2 ) )
  dx0 = x[:nPMaxEf]-x[0]
  candidatos = np.flatnonzero( np.abs(dx0)<tolerancia )[1:]
  while ( len(candidatos)>0 ):
    pP = candidatos[0]
    candidatos = candidatos[1:]
    if np.sum( np.abs(x[:pP]-x[pP:2*pP])>tolerancia )==0:
      return pP
  return naoPeriodico  

for xj, aj in zip(Xs.T, As):
    nPj = contadorPeriodo(xj, nPMax=nPMax, tolerancia=tolnP)
    if retorna_L:
        if nPj > 0:
            dfdx_j = abs(dfdx(xj[-nPj:], aj))
        else:
            dfdx_j = abs(dfdx(xj, aj))
        dfdx_j[dfdx_j < tolLs] = tolLs
        Ls.append(np.mean(np.log(dfdx_j)))