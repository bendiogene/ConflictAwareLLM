import math
import warnings
from functools import partial
from typing import Callable, Iterable, Optional, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from transformers.trainer_utils import SchedulerType
from transformers.utils import logging
from transformers.utils.versions import require_version
import pickle
import os


class AdamWCustomLr(Optimizer):
    """
    Implements a custom version of Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101). This custom optimizer is able to apply a custom lr for each
    neuron

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 1e-3):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        model,
        historical_data_path,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        
    ):
        '''
                if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        '''

        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        
        # Custom checking for historical data file
        _, extension = os.path.splitext(historical_data_path)
        if extension.lower() != '.pkl':
            raise ValueError(f"Invalid .pkl path for historical data: {historical_data_path}")
        
        # Custom message
        print(f'Custom implementation of AdamW optimizer. Loading historical data from {historical_data_path}')
        
        # Extract the historical data: dictionary with structure {'layer_name': Tensor, ...}
        # Each entry contains the name and the lr to be applied to a layer with custom lr requirements
        with open(historical_data_path, 'rb') as file:
            self.historical_data = pickle.load(file)
        
        # Defaults parameters
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}

        
        # Separate names and parameters in order to initialize the optimizer class
        self.names_list = [name for name, _ in model.named_parameters()]
        params_list = [param for _, param in model.named_parameters()]

        # Optimizer object initilization
        super().__init__(params_list, defaults)

        

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # iterate over param groups
        for group in self.param_groups:
            for i,p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                # Substitute the group lr with the custom matrix of lr

                if self.names_list[i] in self.historical_data:
                    step_size = self.historical_data[self.names_list[i]]

                    if group["correct_bias"]:  # No bias correction for Bert
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                    
                    if len(p.data.shape) > 1:
                        # Perform element-wise division and multiplication
                        update = (exp_avg / denom) * (-1) * step_size.unsqueeze(0)
                    else:
                        update = (exp_avg / denom) * (-1) * step_size

                    # Add the result to t in-place
                    p.data.add_(update)     

                else:
                    step_size = group["lr"]

                    if group["correct_bias"]:  # No bias correction for Bert
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                """
                # TO DO: implementation of weight decay for both default lr and custom lr
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=(-group["lr"] * group["weight_decay"]))
                """
                

        return loss
    
    def explain(self):
        for group in self.param_groups:
            # print(len(self.param_groups))
            print(len(group["params"]))

            for name, param in group["params"].named_parameters():
                print(name)
            
            """
            for i,p in enumerate(group["params"]):
                if p.grad is None:
                    print(f'{i} --> MISSING ')
                else:
                    print(f'{i} --> {p.shape}')
            """
            