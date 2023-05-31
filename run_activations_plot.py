import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from fno1d import FNO1d

lw = 2

def set_ax_params(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.grid(linestyle='--' , linewidth=2, alpha=0.5)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

def plot_freqs(ax, act_dicts):
    shift = grid_size // 2
    freqs = torch.arange(0, len(act_dicts[0]['act_fx'])) - shift
    for act_dict in act_dicts:
        ax.plot(freqs, act_dict['fft_act_fx'].abs(),
                label=act_dict['name'], c=act_dict['color'], lw=lw)
    plt.axvline(x=-shannon_freq, linewidth=lw, color='red', linestyle='--', label='Nyquist frequency')
    plt.axvline(x=shannon_freq, linewidth=lw, color='red', linestyle='--')


grid_size = 10000
x_extent = [-math.pi, math.pi]
x = torch.linspace(*x_extent, grid_size)
shannon_freq = grid_size // 500
eps = 1e-1
max_freq = int(shannon_freq * (1 - eps))
fx = sum([5 * np.random.randn() * torch.sin(x * freq) for freq in range(max_freq)]) / max_freq


fno = FNO1d(shannon_freq, 30)
def renorm(x):
    return (x - x.mean()) / x.std()

act_dicts = [
    dict(fn=lambda x: x, color='tab:orange', name='f'),
    dict(fn=F.relu, color='b', name='relu(f)'),
    dict(fn=F.gelu, color='g', name='gelu(f)'),
]
for act_dict in act_dicts:
    act_dict['act_fx'] = act_dict['fn'](fx)
    act_dict['fft_act_fx'] = torch.fft.fftshift(torch.fft.fft(act_dict['act_fx'], norm='forward'))

gs = gridspec.GridSpec(1, 2)
fig = plt.figure(figsize=(12, 10))


plt.subplot(2, 1, 1)

plt.grid()
plt.xlabel('time', fontsize=22)
for act_dict in act_dicts:
    plt.plot(x, act_dict['act_fx'],
             label=f'{act_dict["name"]}(f)',
             c=act_dict["color"], lw=lw)
set_ax_params(plt.gca())
plt.subplot(2, 1, 2)
plt.xlabel('frequency', fontsize=22)
plot_freqs(plt.gca(), act_dicts)
plt.axis([-shannon_freq - 55, shannon_freq + 55, None, None])
plt.ylim(-0.02, 0.36)
set_ax_params(plt.gca())
plt.legend(frameon=False, fontsize=20, bbox_to_anchor=(0.7, 0.85)) # increase legend font size
plt.savefig('spectrum_activations.pdf')

