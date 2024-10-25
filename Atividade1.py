import numpy as np
import matplotlib.pyplot as plt

class Mapa:
    def __init__(self, f, dfdx):
        self.f = f
        self.dfdx = dfdx
    
    def run(self, x0, NT, N, a):
        tamanho_a = len(a) if isinstance(a, (list, np.ndarray)) else 1
        x = np.zeros((N+NT+1, tamanho_a))
        L = np.zeros(tamanho_a)
        x[0,:] = x0
        for i in range(N+NT):
            x[i+1, :] = self.f(x[i], a)
            if i - NT >= 0:
                L += 1/N * np.log(np.abs(self.dfdx(x[i], a)))
        x = x[NT+1:, :]
        return x.T, L
    
    def cobweb(self, nit, x0, a, x_axis = np.linspace(0, 1, 1000), name_save = "Cobweb.png"):
        for ai in a:
            fig = plt.figure(figsize=(10,10))
            x = self.run(x0, 0, nit, ai)[0]
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
            ax.plot(np.ones(100)*x0, np.linspace(0, x[0,0], 100), 'g', linestyle='--')
            ax.plot(np.linspace(x0, x[0,0], 100), np.ones(100)*x[0,0], 'g', linestyle='--')
            for i in range(1,nit):
                ax.plot(np.ones(100)*x[0, i-1], np.linspace(x[0, i-1], x[0, i], 100), 'g', linestyle='--')
                ax.plot(np.linspace(x[0, i-1], x[0,i], 100), np.ones(100)*x[0,i], 'g', linestyle='--')
            plt.savefig(f'a={ai}_{name_save}')
            plt.close()
                
    def serieTemporal(self, x0, NT, N, a, lynestyles = ['.-', '.:', '.:']):
        a = a if isinstance(a, (list, np.ndarray)) else [a]
        for ai in a:
            fig = plt.figure(figsize=(20,10))
            ax = fig.add_subplot(111)
            ax.grid(True)
            ax.set_ylabel('i', fontsize=20)
            ax.set_xlabel('$x_i$', fontsize=20)
            ax.set_title(f'Série temporal para c = {ai}', fontsize=24)
            for i, x0i in enumerate(x0):
                x = self.run(x0i, NT, N, ai)[0]
                ax.plot(range(N), x[0:,].reshape(-1), lynestyles[i], label=f'$x_0 = {x0i}$')
            plt.show()
                  
class DiagramaDeBifurcacao:
    def __init__(self, x, L, a, mapa, setup_reprocess = {"NT": 10_000, "N":1_000}):
        self._ini_a = a
        self._ini_x = x
        self._ini_L = L
        self.a = a
        self.x = x
        self.L = L
        self.mapa = mapa
        self.setup_reprocess = setup_reprocess
        self.define_state()
        
    def define_state(self):
        self.fig = plt.figure(figsize=(10,5))
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
            plt.close(self.fig)
            self.define_state()
            plt.show()
        elif event.key == 'd':
            self.initial_values()
            plt.close(self.fig)
            self.define_state() 
            plt.show()
    
    def on_click(self, event):
        if event.dblclick:
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
        self.ax_Bif.set_title('')
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
        x,L = self.mapa.run(np.random.rand(), **self.setup_reprocess, a = a)
        print("Finalizado")
        self.a = a  
        self.a = a
        self.x = x
        self.L = L
    
    def initial_values(self):
        self.a = self._ini_a
        self.x = self._ini_x
        self.L = self._ini_L
    
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

A = np.linspace(0,2, 4_000)[1:]
mapa = Mapa(**cubico)
x,L = mapa.run(np.random.rand(), 10_000, 1_000, a = A)
diagrama = DiagramaDeBifurcacao(x, L, A, mapa)
diagrama.show()


# xef = mapa.run(np.random.rand(), 100, 1, 0.5)[0][0]

# serieTemporal = {
#     'NT': 0,
#     'N': 100,
#     'a': 1.8,
#     'x0': xef + np.array([0, 1e-15, -1e-15]),
# }
# mapa.serieTemporal(**serieTemporal)