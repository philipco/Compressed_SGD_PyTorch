from math import sqrt

import torch
from torch.optim.optimizer import Optimizer

from quant.quant import prep_grad


class SGDGen(Optimizer):
    r"""
        based on torch.optim.SGD implementation
    """

    def __init__(self, params, step_size, n_workers, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, comp=None, master_comp=None,
                 up_error_feedback=False, down_error_feedback=False, use_up_memory=False, up_compression_model=True,
                 down_compression_model=False):
        if step_size < 0.0:
            raise ValueError("Invalid learning rate: {}".format(step_size))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=step_size, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDGen, self).__init__(params, defaults)

        self.comp = comp
        self.up_error_feedback = up_error_feedback
        self.down_error_feedback = down_error_feedback
        self.use_up_memory = use_up_memory
        self.up_compression_model = up_compression_model
        self.down_compression_model = down_compression_model
        if self.up_error_feedback and self.comp is None:
            raise ValueError("For Error-Feedback, compression can't be None")

        self.master_comp = master_comp  # should be unbiased, Error-Feedback is not supported at the moment

        self.n_workers = n_workers
        self.grads_received = 0

    def __setstate__(self, state):
        super(SGDGen, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step_local_global(self, w_id, closure=None):
        """Performs a single optimization step.

        Arguments:
            w_id: integer, id of the worker
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.grads_received += 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]

                d_p = p.grad.data

                up_error_feedback_name = 'up_error_feedback_' + str(w_id)
                down_error_feedback_name = 'down_error_feedback_' + str(w_id)
                up_memory_name = 'up_memory_' + str(w_id)
                up_learning_rate_name = 'up_learning_rat_' + str(w_id)
                loc_grad = d_p.mul(group['lr'])

                if up_error_feedback_name in param_state:
                    loc_grad += param_state[up_error_feedback_name]  # TODO : multiplier par un coef ?

                if up_memory_name in param_state:
                    loc_grad -= param_state[up_memory_name]

                if self.up_compression_model:
                    d_p = self.comp(loc_grad)
                else:
                    d_p = loc_grad

                if self.up_error_feedback:
                    param_state[up_error_feedback_name] = loc_grad - d_p

                if self.use_up_memory:
                    # Computing learning rate if not already done
                    if up_learning_rate_name not in param_state:
                        _, _, flat_dim = prep_grad(loc_grad)
                        param_state[up_learning_rate_name] = 1 / (2 * ( sqrt(flat_dim) + 1))
                    if up_memory_name in param_state:
                        param_state[up_memory_name] += d_p.mul(param_state[up_learning_rate_name]).detach()
                    else:
                        param_state[up_memory_name] = d_p.mul(param_state[up_learning_rate_name]).detach()

                if 'full_grad' not in param_state or self.grads_received == 1:
                    param_state['full_grad'] = torch.clone(d_p).detach()
                else:
                    param_state['full_grad'] += torch.clone(d_p).detach()

                if self.use_up_memory:
                    param_state['full_grad'] += param_state[up_memory_name].detach()

                if not self.use_up_memory:
                    assert up_memory_name not in param_state, "Up memory should not be in parameters' state."  # torch.equal(param_state['up_global_memory'], torch.zeros_like(param_state['up_global_memory'])), "Global memory must be null."
                if not self.up_error_feedback:
                    assert up_error_feedback_name not in param_state, "Error feedback should not be in parameters' state."

                ###### Computation carried out on  the global server's side. ######
                if self.grads_received == self.n_workers:
                    full_grad = param_state['full_grad'] / self.n_workers

                    if down_error_feedback_name in param_state:
                        full_grad += param_state[down_error_feedback_name]

                    if self.down_compression_model:
                        grad = self.master_comp(full_grad)
                    else:
                        grad = full_grad

                    if self.down_error_feedback:
                        param_state[down_error_feedback_name] = full_grad - grad

                    if weight_decay != 0:
                        grad.add(p, alpha=weight_decay)
                    if momentum != 0:
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                        if nesterov:
                            grad = grad.add(buf, alpha=momentum)
                        else:
                            grad = buf

                    p.data.add_(grad, alpha=-1)

        if self.grads_received == self.n_workers:
            self.grads_received = 0

        return loss
