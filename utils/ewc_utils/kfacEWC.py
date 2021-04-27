import numpy as np
from typing import Dict, Iterable, List, Optional, Tuple
from copy import deepcopy
import torch
import math
from torch import nn
from torch.nn import functional as F

from utils.torch_kfac.layers import init_fisher_block, FisherBlock
from utils.torch_kfac.utils import Lock, inner_product_pairs, scalar_product_pairs


class KfacEWC(object):
    def __init__(self,
                 model: torch.nn.Module,                 
                 device = 'cuda:0', 
                 decay = 0.75,
                 weight = 0.25,
                 center: bool = False) -> None:
        """Creates the KFAC Optimizer object.

        Args:
            model (torch.nn.Module): A `torch.nn.Module` to optimize.
            device (string): the device to run the model on.
            decay (float): the decay parameter for Hessian of the past tasks.
            weight (float): the weight parameter for Hessian of the new task.
                Note that when weight = alpha, decay = 1-alpha, it is an approximation
                to the moving average regime (Chaundry et al., 2018);
                when weight = 1, decay = lambda, it is an approximation to the 
                original EWC-KFAC paper (Ritter et al., 2018) and the Progress & Compress 
                regime (Schwarz et al., 2018).
                Note that in both cases, it is only approximation to the papers
                above, since we are performing online learning. See (Ritter et al.,
                2018, Neurips version, Section 2.5) for details.
            center (bool, optional): If set to True the activations and sensitivities
                are centered. This is useful when dealing with unnormalized distributions.
                Defaults to False.
        """

        self.model = model
        self.device = device
        self.decay = decay
        self.weight = weight
        
        self.blocks: List[FisherBlock] = []
        self.current_blocks: List[FisherBlock] = []

        self.track_forward = Lock()
        self.track_backward = Lock()
        for module in model.modules():
            self.blocks.append(
                init_fisher_block(
                    deepcopy(module),
                    center=center,
                    forward_lock=None,
                    backward_lock=None
                )
            )
        for module in model.modules():
            self.current_blocks.append(
                init_fisher_block(
                    module,
                    center=center,
                    forward_lock=self.track_forward,
                    backward_lock=self.track_backward
                )
            )
            
    def _multiply(self, grads_and_layers: Iterable[Tuple[Iterable[torch.Tensor], FisherBlock]]) -> Iterable[Tuple[Iterable[torch.Tensor], FisherBlock]]:
        return tuple((layer.multiply(grads, damping=0), layer) for (grads, layer) in grads_and_layers)
    
    def reset_cov(self) -> None:
        for block in self.blocks:
            block.reset()

    def reset_current_cov(self) -> None:
        for block in self.current_blocks:
            block.reset()

    def update_current_cov(self, decay=1, weight=1) -> None:
        for layer in self.current_blocks:
            layer.update_cov(cov_ema_decay=decay, weight=weight)
            
    def consolidate(self):
        ''' Consolidate
          This function updates the approximated (empirical) Fisher Information
          Matrix (FIM) to preserve the max log-likelihood for the data in dataloader.
        '''
        for i in range(len(self.blocks)):
            layer = self.blocks[i]
            current_layer = self.current_blocks[i]
            
            # Moving average
            layer._activations_cov.add_to_average(current_layer._activations_cov.value, 
                                                  decay=self.decay, 
                                                  weight=self.weight)
            layer._sensitivities_cov.add_to_average(current_layer._sensitivities_cov.value, 
                                                    decay=self.decay, 
                                                    weight=self.weight)
            
            # Record anchored weights
            layer.module = deepcopy(current_layer.module).to(self.device)


    def update_grad_penalty(self, importance=1):
        ''' Generate the gradient of the approximated online EWC penalty.
            This function receives the current model with its weights, and calculates
            the approximated online EWC loss.
        '''
        # Get grads
        vars_and_layers = []
        current_layers = []
        for i in range(len(self.blocks)):
            layer = self.blocks[i]
            current_layer = self.current_blocks[i]
            if any(grad is not None for grad in current_layer.grads):
                vars_and_layers.append(
                    ([current_layer.vars[j] - layer.vars[j] 
                      for j in range(len(layer.vars))], layer))
                current_layers.append(current_layer)

        # Multiply
        raw_updates_and_layers = self._multiply(vars_and_layers)
        
        # Apply the new gradients
        for i in range(len(raw_updates_and_layers)):
            precon_grad = raw_updates_and_layers[i][0]
            current_layer = current_layers[i]
            current_layer.add_gradients(precon_grad, alpha=importance)

    @property
    def covariances(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [
            (
                block._activations_cov._var,
                block._sensitivities_cov._var
            )
            for block in self.blocks
            if not block.is_static
        ]

    @covariances.setter
    def covariances(self, new_covariances: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        for block, (a_cov, s_cov) in zip(filter(lambda a: not a.is_static, self.blocks), new_covariances):
            block._activations_cov.value = a_cov.to(
                block._activations_cov._var, non_blocking=True)
            block._sensitivities_cov.value = s_cov.to(
                block._sensitivities_cov._var, non_blocking=True)
    
    @property
    def current_covariances(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [
            (
                block._activations_cov._var,
                block._sensitivities_cov._var
            )
            for block in self.current_blocks
            if not block.is_static
        ]

    @current_covariances.setter
    def current_covariances(self, new_covariances: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        for block, (a_cov, s_cov) in zip(filter(lambda a: not a.is_static, self.current_blocks), new_covariances):
            block._activations_cov.value = a_cov.to(
                block._activations_cov._var, non_blocking=True)
            block._sensitivities_cov.value = s_cov.to(
                block._sensitivities_cov._var, non_blocking=True)


