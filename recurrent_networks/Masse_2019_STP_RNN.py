# -*- coding: utf-8 -*-
# %% [markdown]
# # *(Masse, et al., 2019)*: RNN with STP for Working Memory 
# %% [markdown]
# Re-implementation of the paper with BrainPy:
#
# - Masse, Nicolas Y., Guangyu R. Yang, H. Francis Song,
#   Xiao-Jing Wang, and David J. Freedman. "Circuit mechanisms for
#   the maintenance and manipulation of information in working
#   memory." Nature neuroscience 22, no. 7 (2019): 1159-1167.
#
# Thanks the corresponding GitHub code: https://github.com/nmasse/Short-term-plasticity-RNN
#
# The code for the implmentation of Task please refer to the [Masse_2019_STP_RNN_tasks.py](https://github.com/PKU-NIP-Lab/BrainPyExamples/blob/main/recurrent_networks/Masse_2019_STP_RNN_tasks.py).
#
# The analysis methods please refer to the original repository: https://github.com/nmasse/Short-term-plasticity-RNN/blob/master/analysis.py

# %%
import brainpy as bp
import brainpy.math as bm
bp.math.set_platform('cpu')

# %%
import os
import math
import pickle
import numpy as np
from Masse_2019_STP_RNN_tasks import Task

# %%
# Time parameters
dt = 100  # ms
dt_sec = dt / 1000
time_constant = 100  # ms
alpha = dt / time_constant

# %%
# Loss parameters
spike_regularization = 'L2'  # 'L1' or 'L2'
spike_cost = 2e-2
weight_cost = 0.
clip_max_grad_val = 0.1

# %%
# Training specs
batch_size = 1024
learning_rate = 2e-2


# %%
def initialize(shape, prob, size):
  w = bm.random.gamma(shape, size=size)
  w *= (bm.random.random(size) < prob)
  return bm.asarray(w, dtype=bm.float32)


# %% [markdown]
# ## Model

# %%
class Model(bp.layers.Module):
  def __init__(self, task, num_hidden=100, name=None):
    super(Model, self).__init__(name=name)

    assert isinstance(task, Task)
    self.task = task

    # Network configuration
    self.exc_inh_prop = 0.8  # excitatory/inhibitory ratio
    self.conn_prob = 0.2

    # Network shape
    self.num_output = task.num_output
    self.num_hidden = num_hidden
    self.num_input = task.num_input

    # EI
    self.num_exc = int(self.num_hidden * self.exc_inh_prop)
    self.num_inh = self.num_hidden - self.num_exc
    self.EI_list = bm.ones(self.num_hidden)
    self.EI_list[self.num_exc:] = -1.
    self.EI_matrix = bm.diag(self.EI_list)
    self.inh_index = bm.arange(self.num_exc, self.num_hidden)

    # Input and noise
    self.noise_rnn = math.sqrt(2 * alpha) * 0.5

    # Synaptic plasticity specs
    self.tau_fast = 200  # ms
    self.tau_slow = 1500  # ms
    self.U_stf = 0.15
    self.U_std = 0.45

    # Initial hidden values
    self.init_h = bm.TrainVar(bm.ones((batch_size, self.num_hidden)) * 0.1)
    self.h = bm.Variable(bm.ones((batch_size, self.num_hidden)) * 0.1)

    # Input/recurrent/output weights
    #   1. w_ir (input => recurrent)
    prob = self.conn_prob * task.num_receptive_fields
    self.w_ir = bm.TrainVar(initialize(0.2, prob, (self.num_input, self.num_hidden)))
    self.w_ir_mask = bm.ones((self.num_input, self.num_hidden))
    if task.trial_type == 'location_DMS':
      self.w_ir_mask *= 0.
      target_ind = [range(0, self.num_hidden, 3), range(1, self.num_hidden, 3), range(2, self.num_hidden, 3)]
      for n in range(self.num_input):
        u = int(n // (self.num_input / 3))
        self.w_ir_mask[n, target_ind[u]] = 1.
      self.w_ir *= self.w_ir_mask  # only preserve
    #   2. w_rr (recurrent => recurrent)
    self.w_rr = bm.TrainVar(initialize(0.1, self.conn_prob, (self.num_hidden, self.num_hidden)))
    self.w_rr[:, self.num_exc:] = initialize(0.2, self.conn_prob, (self.num_hidden, self.num_inh))
    self.w_rr[self.num_exc:, :] = initialize(0.2, self.conn_prob, (self.num_inh, self.num_hidden))
    self.w_rr_mask = bm.ones((self.num_hidden, self.num_hidden)) - bm.eye(self.num_hidden)
    self.w_rr *= self.w_rr_mask  # remove self-connections
    self.b_rr = bm.TrainVar(bm.zeros((1, self.num_hidden)))
    #   3. w_ro (input => recurrent)
    self.w_ro = bm.TrainVar(initialize(0.1, self.conn_prob, (self.num_hidden, self.num_output)))
    self.w_ro_mask = bm.ones((self.num_hidden, self.num_output))
    self.w_ro_mask[self.num_exc:, :] = 0.
    self.w_ro *= self.w_ro_mask  # remove inhibitory-to-output connections
    #   4. b_ro (bias)
    self.b_ro = bm.TrainVar(bm.zeros((1, self.num_output)))

    # Synaptic variables
    #   - The first row (first half neurons) are facilitating synapses
    #   - The second row (last half neurons) are depressing synapses
    alpha_stf = bm.ones((2, int(self.num_hidden / 2)))
    alpha_stf[0] = dt / self.tau_slow
    alpha_stf[1] = dt / self.tau_fast
    alpha_std = bm.ones((2, int(self.num_hidden / 2)))
    alpha_std[0] = dt / self.tau_fast
    alpha_std[1] = dt / self.tau_slow
    U = bm.ones((2, int(self.num_hidden / 2)))
    U[0] = 0.15
    U[1] = 0.45
    u = bm.ones((batch_size, 2, int(self.num_hidden / 2))) * 0.3
    u[:, 0] = 0.15
    u[:, 1] = 0.45
    #   - final
    self.alpha_stf = alpha_stf.reshape((1, -1))
    self.alpha_std = alpha_std.reshape((1, -1))
    self.U = U.reshape((1, -1))
    self.u = bm.Variable(u.reshape((batch_size, -1)))
    self.x = bm.Variable(bm.ones((batch_size, self.num_hidden)))
    self.y = bm.Variable(bm.ones((batch_size, self.num_output)))
    self.y_hist = bm.Variable(bm.zeros((task.num_steps, batch_size, task.num_output)))

    # Loss
    self.loss = bm.Variable(bm.zeros(1))
    self.perf_loss = bm.Variable(bm.zeros(1))
    self.spike_loss = bm.Variable(bm.zeros(1))
    self.weight_loss = bm.Variable(bm.zeros(1))

  def reset(self):
    u = bm.ones((batch_size, 2, int(self.num_hidden / 2))) * 0.3
    u[:, 0] = 0.15
    u[:, 1] = 0.45
    self.u.value = u.reshape((batch_size, -1))
    self.x.value = bm.ones((batch_size, self.num_hidden))
    self.loss[:] = 0.
    self.perf_loss[:] = 0.
    self.spike_loss[:] = 0.
    self.weight_loss[:] = 0.

  def update(self, input, **kwargs):
    # update STP variables
    self.x += (self.alpha_std * (1 - self.x) - dt_sec * self.u * self.x * self.h)
    self.u += (self.alpha_stf * (self.U - self.u) + dt_sec * self.U * (1 - self.u) * self.h)
    self.x.value = bm.minimum(1., bm.relu(self.x))
    self.u.value = bm.minimum(1., bm.relu(self.u))
    h_post = self.u * self.x * self.h

    # Update the hidden state. Only use excitatory projections from input layer to RNN
    # All input and RNN activity will be non-negative
    state = alpha * (input @ bm.relu(self.w_ir) + h_post @ self.w_rr + self.b_rr)
    state += bm.random.normal(0, self.noise_rnn, self.h.shape)
    self.h.value = bm.relu(state) + self.h * (1 - alpha)
    self.y.value = self.h @ bm.relu(self.w_ro) + self.b_ro

  def predict(self, inputs):
    self.h[:] = self.init_h
    scan = bm.make_loop(body_fun=self.update,
                        dyn_vars=[self.x, self.u, self.h, self.y],
                        out_vars=[self.y, self.h])
    logits, hist_h = scan(inputs)
    self.y_hist[:] = logits
    return logits, hist_h

  def loss_func(self, inputs, targets, mask):
    logits, hist_h = self.predict(inputs)

    # Calculate the performance loss
    perf_loss = bp.losses.cross_entropy_loss(logits, targets, reduction='none') * mask
    self.perf_loss[:] = bm.mean(perf_loss)

    # L1/L2 penalty term on hidden state activity to encourage low spike rate solutions
    n = 2 if spike_regularization == 'L2' else 1
    self.spike_loss[:] = bm.mean(hist_h ** n)
    self.weight_loss[:] = bm.mean(bm.relu(self.w_rr) ** n)

    # final loss
    self.loss[:] = self.perf_loss + spike_cost * self.spike_loss + weight_cost * self.weight_loss
    return self.loss.mean()


# %% [markdown]
# ## Analysis

# %%
def get_perf(target, output, mask):
  """Calculate task accuracy by comparing the actual network output to the desired output
    only examine time points when test stimulus is on, e.g. when y[:,:,0] = 0 """
  target = target.numpy()
  output = output.numpy()
  mask = mask.numpy()

  mask_full = mask > 0
  mask_test = mask_full * (target[:, :, 0] == 0)
  mask_non_match = mask_full * (target[:, :, 1] == 1)
  mask_match = mask_full * (target[:, :, 2] == 1)
  target_max = np.argmax(target, axis=2)
  output_max = np.argmax(output, axis=2)

  match = target_max == output_max
  accuracy = np.sum(match * mask_test) / np.sum(mask_test)
  acc_non_match = np.sum(match * np.squeeze(mask_non_match)) / np.sum(mask_non_match)
  acc_match = np.sum(match * np.squeeze(mask_match)) / np.sum(mask_match)
  return accuracy, acc_non_match, acc_match

# %% [markdown]
# ## Training


# %%
def trial(task_name, save_fn=None, num_iterations=2000, iter_between_outputs=5):
  task = Task(task_name, dt=dt, tau=time_constant, batch_size=batch_size)
  # trial_info = task.generate_trial(set_rule=None)
  # task.plot_neural_input(trial_info)

  model = Model(task)
  opt = bp.optimizers.Adam(learning_rate, train_vars=model.train_vars())
  grad_f = bm.grad(model.loss_func, 
                   dyn_vars=model.vars(),
                   grad_vars=model.train_vars(),
                   return_value=True)

  @bm.jit
  @bp.math.function(nodes=(model, opt))
  def train_op(x, y, mask):
    grads, _ = grad_f(x, y, mask)
    capped_gs = dict()
    for key, grad in grads.items():
      if 'w_rr' in key: grad *= model.w_rr_mask
      elif 'w_ro' in key: grad *= model.w_ro_mask
      elif 'w_ri' in key: grad *= model.w_ir_mask
      capped_gs[key] = bm.clip_by_norm(grad, clip_max_grad_val)
    opt.update(grads=capped_gs)

  # keep track of the model performance across training
  model_performance = {'accuracy': [], 'loss': [], 'perf_loss': [],
                       'spike_loss': [], 'weight_loss': [], 'iteration': []}

  for i in range(num_iterations):
    model.reset()
    # generate batch of batch_train_size
    trial_info = task.generate_trial(set_rule=None)
    inputs = bm.array(trial_info['neural_input'], dtype=bm.float32)
    targets = bm.array(trial_info['desired_output'], dtype=bm.float32)
    mask = bm.array(trial_info['train_mask'], dtype=bm.float32)

    # Run the model
    train_op(inputs, targets, mask)

    # get metrics
    accuracy, _, _ = get_perf(targets, model.y_hist, mask)
    model_performance['accuracy'].append(accuracy)
    model_performance['loss'].append(model.loss)
    model_performance['perf_loss'].append(model.perf_loss)
    model_performance['spike_loss'].append(model.spike_loss)
    model_performance['weight_loss'].append(model.weight_loss)
    model_performance['iteration'].append(i)

    # Save the network model and output model performance to screen
    if i % iter_between_outputs == 0:
      print(task_name +
            f' Iter {i:4d}' +
            f' | Accuracy {accuracy:0.4f}' +
            f' | Perf loss {model.perf_loss[0]:0.4f}' +
            f' | Spike loss {model.spike_loss[0]:0.4f}' +
            f' | Weight loss {model.weight_loss[0]:0.4f}' +
            f' | Mean activity {bm.mean(model.h):0.4f}')

  if save_fn:
    if not os.path.exists(os.path.dirname(save_fn)):
      os.makedirs(os.path.dirname(save_fn))

    # Save model and results
    weights = model.train_vars().unique().dict()
    results = {'weights': weights, 'parameters': {}}
    for k, v in model_performance.items():
      results[k] = v
    pickle.dump(results, open(save_fn, 'wb'))


# %%
trial('DMS')
