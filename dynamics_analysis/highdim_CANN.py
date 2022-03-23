# -*- coding: utf-8 -*-
# %% [markdown]
# # Continuous-attractor Neural Network
# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# %%
import brainpy as bp
import brainpy.math as bm

bm.set_platform('cpu')


# %% [markdown]
# ## Model

# %%
class CANN1D(bp.dyn.NeuGroup):
  def __init__(self, num, tau=1., k=8.1, a=0.5, A=10., J0=4.,
               z_min=-bm.pi, z_max=bm.pi, **kwargs):
    super(CANN1D, self).__init__(size=num, **kwargs)

    # parameters
    self.tau = tau  # The synaptic time constant
    self.k = k  # Degree of the rescaled inhibition
    self.a = a  # Half-width of the range of excitatory connections
    self.A = A  # Magnitude of the external input
    self.J0 = J0  # maximum connection value

    # feature space
    self.z_min = z_min
    self.z_max = z_max
    self.z_range = z_max - z_min
    self.x = bm.linspace(z_min, z_max, num)  # The encoded feature values
    self.rho = num / self.z_range  # The neural density
    self.dx = self.z_range / num  # The stimulus density

    # variables
    self.u = bm.Variable(bm.zeros(num))
    self.input = bm.Variable(bm.zeros(num))

    # The connection matrix
    self.conn_mat = self.make_conn(self.x)

    # function
    self.integral = bp.odeint(self.derivative)

  def derivative(self, u, t, Iext):
    r1 = bm.square(u)
    r2 = 1.0 + self.k * bm.sum(r1)
    r = r1 / r2
    Irec = bm.dot(self.conn_mat, r)
    du = (-u + Irec + Iext) / self.tau
    return du

  def dist(self, d):
    d = bm.remainder(d, self.z_range)
    d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
    return d

  def make_conn(self, x):
    assert bm.ndim(x) == 1
    x_left = bm.reshape(x, (-1, 1))
    x_right = bm.repeat(x.reshape((1, -1)), len(x), axis=0)
    d = self.dist(x_left - x_right)
    Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (bm.sqrt(2 * bm.pi) * self.a)
    return Jxx

  def get_stimulus_by_pos(self, pos):
    return self.A * bm.exp(-0.25 * bm.square(self.dist(self.x - pos) / self.a))

  def update(self, _t, _dt):
    self.u[:] = self.integral(self.u, _t, self.input)
    self.input[:] = 0.

  def cell(self, u):
    return self.derivative(u, 0., 0.)


# %% [markdown]
# ## Helper functions

# %%
def find_fixed_points(cann, do_pca=False, do_animation=False, tolerance=1e-8):
  candidates = cann.get_stimulus_by_pos(bm.arange(-bm.pi, bm.pi, 0.005).reshape((-1, 1)))

  finder = bp.analysis.SlowPointFinder(f_cell=cann.cell)
  finder.find_fps_with_gd_method(
    candidates=candidates,
    tolerance=1e-6, num_batch=200,
    optimizer=bp.optim.Adam(lr=bp.optim.ExponentialDecay(0.1, 2, 0.999)),
  )
  finder.filter_loss(tolerance)
  finder.keep_unique()
  # finder.exclude_outliers(tolerance=1e1)

  print('Losses of fixed points:')
  print(finder.losses)

  if do_pca:
    pca = PCA(2)
    fp_pcs = pca.fit_transform(finder.fixed_points)
    plt.plot(fp_pcs[:, 0], fp_pcs[:, 1], 'x', label='fixed points')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('Fixed points PCA')
    plt.legend()
    plt.show()

  if do_animation:
    bp.visualize.animate_1D(
      dynamical_vars={'ys': finder.fixed_points, 'xs': cann.x, 'legend': 'fixed point'},
      frame_step=1, frame_delay=100, show=True,
    )

  return finder.fixed_points


# %%
def verify_fp_through_simulation(cann, fixed_points, num=3):
  for i in range(num):
    cann.u[:] = fixed_points[i]
    runner = bp.StructRunner(cann, monitors=['u'], dyn_vars=cann.vars())
    runner(100.)
    plt.plot(runner.mon.ts, runner.mon.u.max(axis=1))
    plt.ylim(0, runner.mon.u.max() + 1)
    plt.show()


# %%
def verify_fixed_point_stability(cann, fixed_points, num=3):
  finder = bp.analysis.SlowPointFinder(f_cell=cann.cell)
  J = finder.compute_jacobians(fixed_points[:num])

  for i in range(num):
    eigval, eigvec = np.linalg.eig(np.asarray(J[i]))
    plt.figure()
    plt.scatter(np.real(eigval), np.imag(eigval))
    plt.plot([0, 0], [-1, 1], '--')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.show()


# %%
def visualize_fixed_points(fps, plot_ids=(0,), xs=None):
  for i in plot_ids:
    if xs is None:
      plt.plot(fps[i], label=f'FP-{i}')
    else:
      plt.plot(xs, fps[i], label=f'FP-{i}')
  plt.legend()
  plt.show()


# %% [markdown]
# ## Find fixed points

# %%
model = CANN1D(num=512, k=0.1, A=30, a=0.5)

# %%
fps = find_fixed_points(model, do_pca=True, do_animation=False)

# %%
# verify_fp_through_simulation(model, fps)

# %%
visualize_fixed_points(fps, plot_ids=(10, 20, 30, 40, 50, 60, 70), xs=model.x)

# %%
verify_fixed_point_stability(model, fps, num=6)
