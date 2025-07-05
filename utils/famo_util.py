import torch
from collections import defaultdict


def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == "depth":
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    if task_type == "normal":
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    return loss


def extract_weight_method_parameters_from_args(update_weights_every=1, nashmtl_optim_niter=20, max_norm=1.0,
                                               main_task=0, dwa_temp=2.0, c=0.4, gamma=0.01, method_params_lr=0.025):
    weight_methods_parameters = defaultdict(dict)
    weight_methods_parameters.update(
        dict(
            nashmtl=dict(
                update_weights_every=update_weights_every,
                optim_niter=nashmtl_optim_niter,
                max_norm=max_norm,
            ),
            stl=dict(main_task=main_task),
            dwa=dict(temp=dwa_temp),
            cagrad=dict(c=c, max_norm=max_norm),
            log_cagrad=dict(c=c, max_norm=max_norm),
            famo=dict(gamma=gamma,
                      w_lr=method_params_lr,
                      max_norm=max_norm),
        )
    )
    return weight_methods_parameters