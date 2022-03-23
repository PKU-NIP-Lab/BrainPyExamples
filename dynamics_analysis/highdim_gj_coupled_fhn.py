# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: brainpy
#     language: python
#     name: brainpy
# ---

# %% [markdown]
# # Gap junction-coupled FitzHugh-Nagumo Model

# %%
import brainpy as bp
import brainpy.math as bm

bp.math.enable_x64()
bp.math.set_platform('cpu')

# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
class GJCoupledFHN(bp.dyn.DynamicalSystem):
  def __init__(self, num=2, method='exp_auto'):
    super(GJCoupledFHN, self).__init__()

    # parameters
    self.num = num
    self.a = 0.7
    self.b = 0.8
    self.tau = 12.5
    self.gjw = 0.0001

    # variables
    self.V = bm.Variable(bm.random.uniform(-2, 2, num))
    self.w = bm.Variable(bm.random.uniform(-2, 2, num))
    self.Iext = bm.Variable(bm.zeros(num))

    # functions
    self.int_V = bp.odeint(self.dV, method=method)
    self.int_w = bp.odeint(self.dw, method=method)

  def dV(self, V, t, w, Iext=0.):
    gj = (V.reshape((-1, 1)) - V).sum(axis=0) * self.gjw
    dV = V - V * V * V / 3 - w + Iext + gj
    return dV

  def dw(self, w, t, V):
    dw = (V + self.a - self.b * w) / self.tau
    return dw

  def update(self, _t, _dt):
    self.V.value = self.int_V(self.V, _t, self.w, self.Iext, _dt)
    self.w.value = self.int_w(self.w, _t, self.V, _dt)
    
  def step(self, vw):
    v, w = bm.split(vw, 2)
    dv = self.dV(v, 0., w, self.Iext)
    dw = self.dw(w, 0., v)
    return bm.concatenate([dv, dw])  


# %%
def analyze_net(num=2, gjw=0.01, Iext=bm.asarray([0., 0.6])):
    assert isinstance(Iext, (int, float)) or (len(Iext) == num)
    
    model = GJCoupledFHN(num)
    model.gjw = gjw
    model.Iext[:] = Iext
    
    # simulation
    runner = bp.dyn.DSRunner(model, monitors=['V'])
    runner.run(300.)
    bp.visualize.line_plot(runner.mon.ts, runner.mon.V, legend='V',
                           plot_ids=list(range(model.num)), show=True)

    # analysis
    finder = bp.analysis.SlowPointFinder(f_cell=model.step)
    finder.find_fps_with_gd_method(
      candidates=bm.random.normal(0., 2., (1000, model.num * 2)),
      tolerance=1e-5, num_batch=200,
        optimizer=bp.optim.Adam(lr=bp.optim.ExponentialDecay(0.05, 1, 0.9999)),
    )
    finder.filter_loss(1e-7)
    finder.keep_unique()

    print('fixed_points: ', finder.fixed_points)
    print('losses:', finder.losses)
    if len(finder.fixed_points):
      jac = finder.compute_jacobians(finder.fixed_points)
      for i in range(len(finder.fixed_points)):
        eigval, eigvec = np.linalg.eig(np.asarray(jac[i]))
        plt.figure()
        plt.scatter(np.real(eigval), np.imag(eigval))
        plt.plot([0, 0], [-1, 1], '--')
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.title(f'FP {i}')
        plt.show()


# %% [markdown]
# ## 4D system

# %%
analyze_net(num=2, gjw=0.1, Iext=bm.asarray([0., 0.6]))

# %%
analyze_net(num=2, gjw=0.1, Iext=bm.asarray([0., 0.1]))

# %% [markdown]
# ## 8D system

# %%
analyze_net(num=4, gjw=0.1, Iext=bm.asarray([0., 0., 0., 0.6]))
