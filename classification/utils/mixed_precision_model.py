
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantization_utils.quant_modules import QuantWeight, QuantAct
from .plot_pareto_frontier import plot_to_file
import logging


def _get_wt_layers(model: nn.Module) -> List[QuantWeight]:
    wt_layers = []
    for layer in model.modules():
        if isinstance(layer, QuantWeight):
            wt_layers.append(layer)

    return wt_layers

def _enable_full_precision(model: nn.Module):
    for layer in model.modules():
        if isinstance(layer, QuantWeight) or isinstance(layer, QuantAct):
            layer.full_precision_flag = True

def _disable_full_precision(model: nn.Module):
    for layer in model.modules():
        if isinstance(layer, QuantWeight) or isinstance(layer, QuantAct):
            layer.full_precision_flag = False

def _calc_sens(P, Q):
    def kl_divergence(P, Q):
        batch_sz = P.size(0)
        kl = (P * (P/Q).log()).sum() / batch_sz
        return kl.item()

    return (kl_divergence(P, Q) + kl_divergence(Q, P)) / 2


def search_mixed_precision(quantized_model, dataloader, model_size_limit_mb : float,
        plot_path,
        weight_candidate=[2,4,8],a=5, b=200, t0 = 10, t=5,
    ):
    """
    NOTE: quantized_model is being modified (from the left to the right) within this function
    Enforce a proper lock to the model if required
    """

    def to_b(bits : int):
        """
        Be cautious of potential overflow
        """
        return round((bits / (model_size_limit_mb * 8 * 1024 * 1024)) * b)

    def from_b_to_mb(b_idx : int):
        return (b_idx / float(b)) * (model_size_limit_mb)
    
    # ensure that the order
    weight_candidate.sort()

    m = len(weight_candidate)
    wt_layers = _get_wt_layers(quantized_model)
    L = len(wt_layers)
    
    quantized_model.eval()
    _enable_full_precision(quantized_model)

    # Get the ground truth with full precision using only the first batch
    gt_output = None
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = batch
            sample_batch = batch.cuda()
            gt_output = quantized_model(sample_batch)
            gt_output = F.softmax(gt_output, dim=1)
            break
    
    _disable_full_precision(quantized_model)

    local_sens = [ [0 for z in range(m)] for _ in range(L)]
    for layer_idx, wt_layer in enumerate(wt_layers):
        for wt_idx, wt_bit in enumerate(weight_candidate):
            wt_layer.set_weight_bit(wt_bit)
            with torch.no_grad():
                output = quantized_model(sample_batch)
                output = F.softmax(output, dim=1)
                local_sens[layer_idx][wt_idx]= _calc_sens(output, gt_output)

    # all the weights are set to the last bit of the 

    ### BEGIN Init
    dp = [ [ Configs() for z in range(b)] for y in range(L+1)]
    dp[-1][0].add(Config(0, [], 0))
    dp[-1][0].sort_and_trim(t0)
    ### END Init

    # Group every `a` layers
    for lgrp_idx in range( (L + a + 1) // a):
        a0 = min(L, (lgrp_idx+1) * a) - lgrp_idx * a

        for layer_idx_in_grp in range(a0):
            
            layer_idx = lgrp_idx * a + layer_idx_in_grp
            logging.info(layer_idx)
            
            wt_layer = wt_layers[layer_idx]

            for wt_idx, wt_bit in enumerate(weight_candidate):
                delta_bit = wt_layer.get_param_size() * wt_bit
                delta_sens = local_sens[layer_idx][wt_idx]

                # we may lose some valid candidates due to an overflow in the next line
                stop_condition = False
                for prv_b in range(b):
                    if stop_condition:
                        break 
                    
                    for config in dp[layer_idx-1][prv_b].configs:
                        now_b = to_b(config.acc_bit + delta_bit)
                        if now_b >= b:
                            stop_condition = True
                            break
                                                
                        dp[layer_idx][now_b].add(
                            Config(
                                sens=config.sens + delta_sens,
                                list_wt_bit=config.list_wt_bit + [wt_bit],
                                acc_bit=config.acc_bit + delta_bit
                            )
                        )
                        dp[layer_idx][now_b].sort_and_trim(t0)
        
        for b_idx in range(b):
            configs = dp[lgrp_idx * a + a0 -1][b_idx].configs
            for config in configs:
                for layer_idx, wt_bit in enumerate(config.list_wt_bit):
                    wt_layers[layer_idx].set_weight_bit(wt_bit)
                
                with torch.no_grad():
                    output = quantized_model(batch)
                    output = F.softmax(output, dim=1)

                    # recalculate the sensitivity without 'independence assumption'
                    config.sens = _calc_sens(output, gt_output)

            dp[lgrp_idx * a + a0 -1][b_idx].sort_and_trim(t)

    ### BEGIN plot
    x = []
    y = []

    x_pareto = []
    y_pareto = []
    
    best_config : Config = None

    for b_idx in range(b):
        if dp[-2][b_idx].configs:
            sens = dp[-2][b_idx].configs[0].sens
            sz = from_b_to_mb(b_idx)
            x_pareto.append(sz)
            y_pareto.append(sens)

            if best_config is None or best_config.sens > sens:
                best_config = dp[-2][b_idx].configs[0]

        for config in dp[-2][b_idx].configs:
            x.append(sz)
            y.append(config.sens)
        
    plot_to_file(x, y, x_pareto, y_pareto, plot_path)
    ### END plot

    # set weight bits with `best_config`
    for layer_idx, wt_layer in enumerate(wt_layers):
        wt_layer.set_weight_bit(best_config.list_wt_bit[layer_idx])

    return best_config 


class Config:
    def __init__(self, sens: float, list_wt_bit : List[int], acc_bit : int):
        # sensitivity
        self.sens = sens

        # list of weight bits
        self.list_wt_bit = list_wt_bit

        # Accumlated bit
        self.acc_bit = acc_bit

    def __repr__(self):
        return f"{{\n\tsens: {self.sens:.3f},\n\tbit : {self.list_wt_bit}\n\tsize: {self.acc_bit / ( 8 * 1024 * 1024):.3f}MB}}"
            
    
class Configs:
    def __init__(self):
       self.configs = []

    def add(self, config : Config):
        self.configs.append(config)

    def sort_and_trim(self, t=10):
        self.configs.sort(key=lambda c: (c.sens, c.acc_bit))
        self.configs = self.configs[:t]