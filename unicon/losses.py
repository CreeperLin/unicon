import torch
import torch.nn.functional as F

DEBUG = False


def chamfer_loss2(pred, true, alpha=0.5, compute_mode='donot_use_mm_for_euclid_dist'):
    # B M N
    if len(pred) > 16384:
        compute_mode = 'use_mm_for_euclid_dist_if_necessary'
    if len(pred) != len(true):
        print(pred.shape, true.shape)
        pred = pred.view(-1, *true.shape)
        true = true.view(-1, *true.shape)
    n = true.shape[-1]
    # print(n)
    # B M M
    dist = torch.cdist(pred, true, p=2, compute_mode=compute_mode)
    dist = dist.view(-1, *dist.shape[-2:])
    # B M
    min_distances1, argmin1 = torch.min(dist, dim=-1)
    min_distances2, argmin2 = torch.min(dist, dim=-2)
    if DEBUG:
        # print('DEBUG')
        chamfer_loss.argmins = [argmin1.clone(), argmin2.clone()]
    loss = (1 - alpha) * torch.mean(min_distances1, dim=-1) + alpha * torch.mean(min_distances2, dim=-1)
    return loss


def chamfer_loss(pred, true, alpha=0.5):
    # B M N
    if len(pred) != len(true):
        print(pred.shape, true.shape)
        pred = pred.view(-1, *true.shape)
        true = true.view(-1, *true.shape)
    n = true.shape[-1]
    # B M M
    dist = (pred.unsqueeze(-3) - true.unsqueeze(-2)).square().mean(dim=-1)
    dist = dist.view(-1, *dist.shape[-2:])
    # B M
    min_distances1, argmin1 = torch.min(dist, dim=-1)
    min_distances2, argmin2 = torch.min(dist, dim=-2)
    if DEBUG:
        # print('DEBUG')
        chamfer_loss.argmins = [argmin1.clone(), argmin2.clone()]
    loss = (1 - alpha) * torch.mean(min_distances1, dim=-1) + alpha * torch.mean(min_distances2, dim=-1)
    return loss


def rmse_loss(pred, true):
    # print('pred', pred[0, :8, 0:6])
    # print('true', true[0, :8, 0:6])
    # return torch.sqrt((true-pred).square().sum(dim=(-1, -2)).mean())
    # return torch.sqrt((true-pred).square().sum(dim=(-2)).mean())
    return torch.sqrt(f_mse_loss(pred, true))


def f_mse_loss(pred, true):
    # return F.mse_loss(pred, true)
    loss = F.mse_loss(pred, true, reduction='none')
    return loss.mean(dim=-1).sum(dim=-1)


def mse_loss(pred, true):
    # return F.mse_loss(pred, true)
    loss = F.mse_loss(pred, true, reduction='none')
    return loss.mean(dim=(-1, -2))


def unfolded_mse_loss(pred, true, step=1, sym=True, alpha=0.5):
    # B M N
    len_pred = pred.shape[1]
    len_true = true.shape[1]
    assert len(pred.shape) == 3
    # W
    w_size = len_pred // 2
    pred_u = torch.unfold_copy(pred, dimension=1, size=w_size, step=step).permute(0, 1, 3, 2)
    # B P W N
    true_w = true[:, :w_size].unsqueeze(1)
    # B 1 W N
    loss1 = torch._C._nn.mse_loss(pred_u, true_w, F._Reduction.get_enum('none'))
    loss1 = loss1.mean(dim=(-1, -2))
    # B P
    # loss1 = loss1.mean(dim=-1).sum(dim=-1)
    # print(loss1)
    min_loss1, argmin1 = torch.min(loss1, dim=-1)
    # print(min_loss1, argmin1)
    if not sym:
        return min_loss1
    true_u = torch.unfold_copy(true, dimension=1, size=w_size, step=step).permute(0, 1, 3, 2)
    pred_w = pred[:, :w_size].unsqueeze(1)
    # loss2 = F.mse_loss(pred_w, true_u, reduction='none')
    loss2 = torch._C._nn.mse_loss(pred_w, true_u, F._Reduction.get_enum('none'))
    loss2 = loss2.mean(dim=(-1, -2))
    # loss2 = loss2.mean(dim=-1).sum(dim=-1)
    if DEBUG:
        # print('DEBUG')
        unfolded_mse_loss.losses = [loss1, loss2]
    # print(loss2)
    # unfolded_mse_loss.loss1 = loss1
    min_loss2, argmin2 = torch.min(loss2, dim=-1)
    # print(min_loss2, argmin2)
    return alpha * min_loss1 + (1 - alpha) * min_loss2
    # return torch.where(min_loss1 < min_loss2, min_loss1, min_loss2)


def plot_unfolded_mse_loss(title=None):
    losses = getattr(unfolded_mse_loss, 'losses', None)
    if losses is not None:
        from matplotlib import pyplot as plt
        for k, loss in enumerate(losses):
            print('umse', k, loss.shape)
            loss = loss.cpu().numpy()
            x = range(loss.shape[-1])
            fig = plt.figure()
            for i in range(len(loss)):
                plt.plot(x, loss[i])
            if title is not None:
                plt.title(title)
            fig.tight_layout()
            # plt.plot(loss1)
            plt.savefig(f'loss_{k}.png')


def plot_chamfer_loss(title=None):
    argmins = getattr(chamfer_loss, 'argmins', None)
    if argmins is not None:
        # print('argmins', argmins)
        from matplotlib import pyplot as plt
        for i, argmin in enumerate(argmins):
            argmin = argmin.cpu()
            L = argmin.shape[1]
            # data = torch.zeros(L, L)
            data = torch.ones(L, L)
            for d in argmin:
                data[torch.arange(0, L), d] = data[torch.arange(0, L), d] + 1
            data = torch.log10(data)
            # print(data)
            fig = plt.figure()
            plt.imshow(data, interpolation='nearest', cmap='hot')
            plt.colorbar()
            if title is not None:
                plt.title(title)
            fig.tight_layout()
            plt.savefig(f'argmin_{i}.png')
