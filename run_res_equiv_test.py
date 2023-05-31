import os
from functools import partial
import argparse

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from reno.fno2d import FNO2d
from reno.sno import TrigonometricSNO
from reno.unet import UnetPermuted
from reno.utilities import rel_norm, interp_fourier_2d
from utilities3 import LpLoss


def compute_discrete_aliasing_error(x, fno, device='cuda'):
    # expecting x: (batch_size, size, size, 1)
    x = x.to(device)
    orig_size1, orig_size2 = x.shape[-3], x.shape[-2]
    assert orig_size1 == orig_size2
    orig_size = orig_size1
    loss = partial(rel_norm, p=1)
    fno_x = fno(x)
    min_sizes_mapper = {
        'fno': 31,
        'sno': 3,
        'unet': 17,
    }
    sizes = [min_sizes_mapper[args.model] + 2 * i for i in range(100)]
    errs = []
    for size in sizes:
        hr_x = interp_fourier_2d(x, (size, size))
        fno_hr_x = fno(hr_x)
        lr_fno_hr_x = interp_fourier_2d(fno_hr_x, (orig_size, orig_size))
        # compute errors on training grid
        errori = loss(lr_fno_hr_x, fno_x)
        errs.append(errori.item())
    plt.plot(sizes, errs)
    plt.show()
    return sizes, errs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default='fno', choices=['fno', 'unet', 'sno'])
    parser.add_argument("-o", "--output_root", type=str, default='')
    parser.add_argument("-e", "--save_at_epoch", type=int, default=499)
    args = parser.parse_args()

    nsamples = 128
    batch_size = 32
    learning_rate = 0.002
    epochs = 500
    s = 61
    torch.manual_seed(0)
    x_train = torch.randn(nsamples, s, s, 1) / 3
    y_train = torch.randn(nsamples, s, s, 1) / 3
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=batch_size, shuffle=True)

    ################################################################
    # training and evaluation
    ################################################################
    if args.model == 'fno':
        model = FNO2d(16, 16, 32).cuda()
    elif args.model == 'unet':
        model = UnetPermuted(1, 1, bilinear=False).cuda()
    elif args.model == 'sno':
        model = TrigonometricSNO(1, 1, (s, s)).cuda()
    else:
        raise NotImplementedError(args.model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    myloss = LpLoss(size_average=False)
    for ep in range(epochs):
        model.train()
        train_l2 = 0
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x).reshape(batch_size, s, s)
            loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

        model.eval()
        eval_l2 = 0.0
        with torch.no_grad():
            errors_grids = []
            for i, (x, y) in enumerate(loader):
                x, y = x.cuda(), y.cuda()

                out = model(x).reshape(batch_size, s, s)
                if ep == args.save_at_epoch:
                    res_grid, errors_grid = compute_discrete_aliasing_error(x, model)
                    errors_grids.append(errors_grid)

                # if ep % 50 == 0 and i == 0:
                #     compute_discrete_aliasing_error(x, model)

                eval_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

            train_l2 /= nsamples
            eval_l2 /= nsamples

            if ep == args.save_at_epoch:
                fn = f'{args.model}_epoch_{ep}_loss_{eval_l2}.csv'
                path = os.path.join(os.getcwd(), args.output_root, fn)
                print(f'Saving results to {path}')
                errors_grid_avg = np.array(errors_grids).mean(0)
                plt.title('saved one')
                plt.plot(res_grid, errors_grid_avg)
                plt.show()
                pd.Series(errors_grid_avg, index=res_grid, name=args.model) \
                    .to_csv(path, index_label="res")

        print(ep, train_l2, eval_l2)
