import numpy as np
import matplotlib.pyplot as plt

class Mapa:
    def __init__(self, f, dfdx):
        self.f = f
        self.dfdx = dfdx
        
    def contadorPeriodo(self, x, nPMax=np.inf, tolerancia=1e-7, naoPeriodico=0):
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
    
    def run(self, x0, NT, N, a):
        tamanho_a = len(a) if isinstance(a, (list, np.ndarray)) else 1
        Xs = np.zeros( (N, tamanho_a) )
        x = x0
        for i in range( -NT, N ):
            x = self.f(x,a)
            if i>=0:
                Xs[i,:] = x
        return Xs.T
    
    def LyapunovNps(self, x0, NT, N, a, tolLs = 1e-15, nPMax = 100, tolnP = 1e-7):
        Xs = self.run(x0, NT, N, a).T
        Ls = []
        nPs = []
        for xj, aj in zip( Xs.T, a):
            nPj = self.contadorPeriodo(xj, nPMax=nPMax, tolerancia=tolnP)
            if nPj>0:
                dfdx_j = abs( self.dfdx(xj[-nPj:],aj) )
            else:
                dfdx_j = abs( self.dfdx(xj,aj) )
            dfdx_j[dfdx_j<tolLs] = tolLs
            Ls.append( np.mean( np.log( dfdx_j ) ) )
            nPs.append( nPj )
        return Xs.T, np.array(Ls), np.array(nPs)
    
    def cobweb(self, nit, x0, a, x_axis = np.linspace(0, 1, 1000), name_save = "Cobweb.png"):
        for ai in a:
            fig = plt.figure(figsize=(10,10))
            x = self.run(x0, 0, nit, ai)
            #Definições básicas do gráfico
            ax = fig.add_subplot(111)
            ax.grid(True)
            ax.set_ylabel('f(x)', fontsize=20)
            ax.set_xlabel('x', fontsize=20)
            ax.set_title(f'Cobweb para c = {ai}', fontsize=24)
            #Plota a função
            ax.plot(x_axis, self.f(x_axis, ai), 'r')
            #Plota a reta y = x
            ax.plot(x_axis, x_axis, 'b')
            #Plota primeiro ponto
            ax.plot(np.ones(100)*x0, np.linspace(x0, x[0,0], 100), 'g', linestyle='--')
            ax.plot(np.linspace(x0, x[0,0], 100), np.ones(100)*x[0,0], 'g', linestyle='--')
            for i in range(1,nit):
                ax.plot(np.ones(100)*x[0, i-1], np.linspace(x[0, i-1], x[0, i], 100), 'g', linestyle='--')
                ax.plot(np.linspace(x[0, i-1], x[0,i], 100), np.ones(100)*x[0,i], 'g', linestyle='--')
            plt.savefig(f'a={ai}_{name_save}')
            plt.close()
                
    def serieTemporal(self, x0, NT, N, a, lynestyles = ['.-', '.:', '.:'], legenda = []):
        a = a if isinstance(a, (list, np.ndarray)) else [a]
        for ai in a:
            fig = plt.figure(figsize=(20,10))
            ax = fig.add_subplot(111)
            ax.grid(True)
            ax.set_ylabel('$x_i$', fontsize=20)
            ax.set_xlabel('$i$', fontsize=20)
            ax.set_title(f'Série temporal para c = {ai}', fontsize=24)
            for i, x0i in enumerate(x0):
                x = self.run(x0i, NT, N, ai)
                ax.plot(range(N), x[0:,].reshape(-1), lynestyles[i], label=legenda[i])
            ax.legend(loc = 'lower left', fontsize=20)
            plt.show()
                  
class DiagramaDeBifurcacao:
    def __init__(self, x, L, nPs, a, mapa, setup_reprocess = {"NT": 10_000, "N":1_000, "tolLs": 1e-15, "nPMax": 32, "tolnP": 1e-7}):
        self._ini_a = a
        self._ini_x = x
        self._ini_L = L
        self._ini_nPs = nPs
        self.a = a
        self.x = x
        self.L = L
        self.nPs = nPs
        self.mapa = mapa
        self.setup_reprocess = setup_reprocess
        self.fig = plt.figure(figsize=(20,10))
        self.define_state()
        
    def define_state(self):
        self.fig.clear()
        self.ax_Bif = self.fig.add_subplot(211)
        self.ax_Lyap = self.fig.add_subplot(212)
        self.build_default_state(self.ax_Bif, y_label = "x")
        self.build_default_state(self.ax_Lyap, y_label = r"$\lambda$", x_label = "c")
        self.ax_Bif.plot(self.a, self.x, ',k', alpha=0.2)
        self.ax_Lyap.plot(self.a, self.L, 'g')
        self.cid1 = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.cid2 = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.show_Lyap = True
        self.show_Bif = True
        self.show_periodos = False
        self.ax_Bif.set_title("Diagrama de Bifurcação", fontsize=24)
        self.ax_Bif.callbacks.connect('xlim_changed', self.on_xlim_changed(self.ax_Lyap))
        self.ax_Lyap.callbacks.connect('xlim_changed', self.on_xlim_changed(self.ax_Bif))
        
    def on_key(self, event):
        if event.key =='z':
            self.Lyapunov_Only()
            self.fig.canvas.draw_idle()
        elif event.key == 'x':
            self.Bifurcacao_Only()
            self.fig.canvas.draw_idle()
        elif event.key == 'c':
            self.reset_state()
            self.fig.canvas.draw_idle()
        elif event.key == 'a':
            self.reprocess()
            self.define_state()
            self.fig.canvas.draw_idle()
        elif event.key == 'd':
            self.initial_values()
            self.define_state() 
            self.fig.canvas.draw_idle()
        elif event.key == 'g':
            self.plotaPeriodos()
            self.fig.canvas.draw_idle()
    
    def on_click(self, event):
        if event.button == 2:
            print(event.xdata, event.ydata)
    
    def build_default_state(self, ax, y_label, x_label = "", fontsize=20):
        ax.grid(True)
        ax.set_ylabel(y_label, fontsize=20)
        if x_label:
            ax.set_xlabel(x_label, fontsize=20)
        
    def Lyapunov_Only(self):
        self.show_Lyap = True
        if self.show_Lyap and not self.show_Bif :
            return
        self.ax_Bif.set_visible(False)
        self.ax_Lyap.set_visible(True)
        self.ax_Lyap.set_position([0.1, 0.1, 0.8, 0.8])
        self.ax_Lyap.set_xlabel('c', fontsize=20)
        self.ax_Lyap.set_title(f'Expoente de Lyapunov', fontsize=24)
        self.show_Bif = False
        
    def Bifurcacao_Only(self):
        self.show_Bif = True
        if self.show_Bif and not self.show_Lyap:
            return
        self.ax_Lyap.set_visible(False)
        self.ax_Bif.set_visible(True)
        self.ax_Bif.set_position([0.1, 0.1, 0.8, 0.8])
        self.ax_Bif.set_xlabel('c', fontsize=20)
        self.ax_Bif.set_title(f'Diagrama de Bifurcação', fontsize=24)
        self.show_Lyap = False
              
    def reset_state(self):
        self.show_Lyap = True
        self.show_Bif = True
        self.ax_Bif.set_visible(True)
        self.ax_Lyap.set_visible(True)
        self.ax_Bif.set_position([0.1, 0.5, 0.8, 0.4])
        self.ax_Lyap.set_position([0.1, 0.1, 0.8, 0.4])
        self.ax_Bif.set_title('Diagrama de Bifurcação', fontsize=24)
        self.ax_Lyap.set_title('')
        self.ax_Lyap.set_xlabel('c', fontsize=20)
        self.ax_Bif.set_xlabel('')
          
    def on_xlim_changed(self, ax):      
        def callback(event_ax):
            if event_ax is not ax:
                # Obter os limites de x dos eixos de origem e destino
                xlim_source = event_ax.get_xlim()
                xlim_target = ax.get_xlim()
                
                # Comparar os limites de x
                if xlim_source != xlim_target:
                    # Atualizar os limites do eixo de destino
                    ax.set_xlim(xlim_source)
                    ax.figure.canvas.draw_idle()
        return callback
        
    def reprocess(self):
        xlim = self.ax_Bif.get_xlim()
        a = np.linspace(xlim[0], xlim[1], len(self.a))
        print("Iterando Mapa")
        x,L,nPs = self.mapa.LyapunovNps(np.random.rand(), **self.setup_reprocess, a = a)
        print("Finalizado")
        self.a = a  
        self.nPs = nPs
        self.x = x
        self.L = L
    
    def initial_values(self):
        self.a = self._ini_a
        self.x = self._ini_x
        self.L = self._ini_L
        self.nPs = self._ini_nPs
    
    def plotaPeriodos(self):
        if not self.show_Lyap:
            return
        if self.show_periodos:
            self.define_state()
            self.show_periodos = False
            return
        self.ax_Lyap.cla()
        #Printa cada periodo com uma cor diferente
        for i in range(1, np.max(self.nPs)+1):
            self.ax_Lyap.plot(self.a[self.nPs == i], self.L[self.nPs == i], '.')
        self.ax_Lyap.set_xlabel('c', fontsize=20)
        self.ax_Lyap.set_ylabel(r'$\lambda$', fontsize=20)
        self.ax_Lyap.grid(True)
        self.show_periodos = True
    
    def show(self):
        plt.show()  
            
logistico = {
    'f': lambda x, a: a*x*(1-x),
    'dfdx': lambda x, a: a*(1-2*x),
}

cubico = {
    'f': lambda x, a: 1 - a*np.abs(x)**3,
    'dfdx': lambda x, a: -3*a*x**2,
}

mapa = Mapa(**cubico)

# # 1 - a) Código usado para a geração das figuras 1, 2, 3 e 5
# A = np.linspace(0,2, 10_000)[1:-1]
# x, L, nPs = mapa.LyapunovNps(np.random.rand(), 100_000, 1_000, A)

# diagrama = DiagramaDeBifurcacao(x, L, nPs, A, mapa, setup_reprocess={"NT": 100_000, "N":1_000, "tolLs": 1e-15, "nPMax": 32, "tolnP": 1e-7})
# diagrama.show()

# # 1 -a) Código usado para a geração das figuras 4 e 6
# xef = mapa.run(np.random.rand(), 100, 1, 0.5).T[0]

# delta0 = 1e-10

# serieTemporal = {
#     'NT': 0,
#     'N': 100,
#     'a': [0.65, 1.89],
#     'x0': xef + np.array([0, delta0, -delta0]),
#     'legenda': ['$x_0$', '$x_0 + 10^{-10}$', '$x_0 - 10^{-10}$'],
# }

# mapa.serieTemporal(**serieTemporal)

#1-a) Código usado para a geração das figuras 7, 8 e 9
cobweb = lambda a:{
    'nit': 15,
    'a': [a],
    'x0': mapa.run(np.random.rand(), 100_000, 1, a).T[0],
    'name_save': 'Cobweb.png',
    'x_axis': np.linspace(-1, 1, 1000),
}

As = [0.65, 1.35, 1.89]
for a in As:
    mapa.cobweb(**cobweb(a))