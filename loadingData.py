import pickle as pkl
import modules.dtset as dtset
import modules.args
import modules.dencoders as endecs
import modules.advecdiffus as warps
import modules.losses as losses
import modules.plots as plot
from modules.meter import AverageMeters


import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.image as mpimg


from netCDF4 import Dataset


ncfile = '/Users/mostafa/Desktop/datas/nnx.nc'
fh = Dataset(ncfile, mode='r')
# th0 = fh.variables['thetao'][:]
# th0 = fh.variables['thetao'][0,0,:,:]
th0 = fh.variables['thetao'][:]



output = open('/Users/mostafa/Desktop/datas/train/data_1.pkl', 'wb')
pkl.dump(th0, output)
output.close()


data = pkl.load(open('/Users/mostafa/Desktop/datas/train/data_1.pkl', 'rb'))

print(data.shape)

num_single = data.shape[0] - 4 - 6 - 1


print(num_single)

sample_num = 100 % (num_single - 1)

print(sample_num)

input = data[sample_num: sample_num + 4]

target = data[sample_num + 4: sample_num + 4 + 6]



print(input)

print(input.shape)


args = modules.args.parser.parse_args()

dset = dtset.Dset(args.train_root,
                  seq_len=args.seq_len,
                  target_seq_len=args.target_seq_len,
                  )

test_dset = dtset.Dset(args.test_root,
                       seq_len=args.seq_len,
                       target_seq_len=args.target_seq_len,
                       )
#
#     train_indices = range(0, int(len(dset[:]) * args.split))
#     val_indices = range(int(len(dset[:]) * args.split), len(dset[:]))
#
#     train_loader = DataLoader(dset,
#                               batch_size=args.batch_size,
#                               sampler=SubsetRandomSampler(train_indices),
#                               )
#
#     val_loader = DataLoader(dset,
#                             batch_size=args.batch_size,
#                             sampler=SubsetRandomSampler(val_indices),
#                             )
#
#     test_loader = DataLoader(test_dset,
#                              batch_size=args.batch_size,
#                              shuffle=False,
#                              )
#
#     splits = {
#         'train': train_loader,
#         'valid': val_loader,
#         'test': test_loader
#     }
#
#     print('>>>> employing the encode-decode network...\n')
#     endecoder = endecs.ConvDeconvEstimator(input_channels=args.seq_len,
#                                            upsample_mode=args.upsample)
#
#     print('>>>> creating warping scheme {}'.format(args.warp))
#     warp = warps.__dict__[args.warp]()
#
#     print('\n>>>> implementing loss function...\n')
#
#     photo_loss = nn.MSELoss()
#     smooth_loss = losses.SmoothnessLoss(photo_loss)
#     div_loss = losses.DivergenceLoss(photo_loss)
#     magn_loss = losses.MagnitudeLoss(photo_loss)
#
#     cudnn.benchmark = True
#     optimizer = optim.Adam(endecoder.parameters(), args.lr,
#                            betas=(args.momentum, args.beta),
#                            weight_decay=args.weight_decay)
#
#     _x, _ys = torch.Tensor(), torch.Tensor()
#     viz_wins = {}
#
#     for epoch in range(1, args.epochs + 1):
#
#         results = {}
#
#         for split, dl in splits.items():
#             print(dl)
#
#             meters = AverageMeters()
#
#             if split == 'train':
#                 endecoder.train(), warp.train()
#             else:
#                 endecoder.eval(), warp.eval()
#
#             for i, (input, targets) in enumerate(dl):
#
#                 _x.resize_(input.size()).copy_(input)
#                 _ys.resize_(targets.size()).copy_(targets)
#                 _ys = _ys.transpose(0, 1).unsqueeze(2)
#                 x, ys = Variable(_x), Variable(_ys)
#
#                 pl = 0
#                 sl = 0
#                 dl = 0
#                 ml = 0
#
#
#                 ims = []
#                 ws = []
#                 last_im = x[:, -1].unsqueeze(1)
#                 for y in ys:
#
#                     # x.permute(3,2,1,0)
#
#
#
#                     # x = mpimg.imread('test.jpg', 0)
#                     # x = torch.tensor(x)
#
#
#                     # print('x', x.shape)
#
#
#                     # w = endecoder(x)
#                     im = warp(x[:, -1].unsqueeze(1), w)
#                     x = torch.cat([x[:, 1:], im], 1)
#
#                     curr_pl = photo_loss(im, y)
#                     pl += torch.mean(curr_pl)
#                     sl += smooth_loss(w)
#                     dl += div_loss(w)
#                     ml += magn_loss(w)
#
#                     ims.append(im.cpu().data.numpy())
#                     ws.append(w.cpu().data.numpy())
#
#                 pl /= args.target_seq_len
#                 sl /= args.target_seq_len
#                 dl /= args.target_seq_len
#                 ml /= args.target_seq_len
#
#                 loss = pl + args.smooth_coef * sl + args.div_coef * dl + args.magn_coef * ml
#
#                 if split == 'train':
#                     optimizer.zero_grad()
#                     # loss.backward()
#                     optimizer.step()
#
#                 # meters.update(
#                 #     dict(loss=loss.data[0],
#                 #          pl=pl.data[0],
#                 #          dl=dl.data[0],
#                 #          sl=sl.data[0],
#                 #          ml=ml.data[0],
#                 #          ),
#                 #     n=x.size(0)
#                 # )
#
#
#     plot.plot(dset)
#
#
#
#
#
