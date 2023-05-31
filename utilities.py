import torch

def fourier_magnitude(x, permute=True, norm='forward', shift=True):
    if permute:
        x = x.permute(0, 3, 1, 2)
    x = torch.fft.fft2(x, norm=norm)
    if shift:
        x = torch.fft.fftshift(x, dim=(-2, -1))
    if permute:
        x = x.permute(0, 2, 3, 1)
    return x.abs()

def rel_norm(out, y, p):
    assert out.dim() == 4
    assert y.dim() == 4
    assert p > 0
    reduce = lambda x : x.sum(-1).sum(-1).sum(-1)
    l = reduce(abs(out - y) ** p) / reduce(abs(y) ** p)
    assert len(l.shape) == 1
    return l.mean() * 100

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

def interp_fourier_2d(x, out_size, permute=True):
    if permute:
        x = x.permute(0, 3, 1, 2)

    f = torch.fft.rfft2(x, norm='backward')
    f_z = torch.zeros((*x.shape[:-2], out_size[0], out_size[1]//2 + 1), dtype=f.dtype, device=f.device)
    # 2k+1 -> (2k+1 + 1) // 2 = k+1 and (2k+1)//2 = k
    top_freqs1 = min((f.shape[-2] + 1) // 2, (out_size[0] + 1) // 2)
    top_freqs2 = min(f.shape[-1], out_size[1] // 2 + 1)
    # 2k -> (2k + 1) // 2 = k and (2k)//2 = k
    bot_freqs1 = min(f.shape[-2] // 2, out_size[0] // 2)
    bot_freqs2 = min(f.shape[-1], out_size[1] // 2 + 1)
    f_z[..., :top_freqs1, :top_freqs2] = f[..., :top_freqs1, :top_freqs2]
    f_z[..., -bot_freqs1:, :bot_freqs2] = f[..., -bot_freqs1:, :bot_freqs2]
    x_z = torch.fft.irfft2(f_z, s=out_size).real
    x_z = x_z * (out_size[0] / x.shape[-2]) * (out_size[1] / x.shape[-1])

    # f_z[..., -f.shape[-2]//2:, :f.shape[-1]] = f[..., :f.shape[-2]//2+1, :]

    if permute:
        x_z = x_z.permute(0, 2, 3, 1)
    return x_z