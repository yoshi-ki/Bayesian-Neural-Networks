from __future__ import division, print_function
import time
import torch.utils.data
from torchvision import transforms, datasets
import argparse
import matplotlib
from src.Bayes_By_Backprop.act_test_VGG11 import *

matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Test Bayesian Convolutional Neural Net on cifar10 with Variational Inference')
parser.add_argument('n_samples', type=float, nargs='?', action='store', default=10,
                    help='How many MC samples to take when approximating the ELBO. Default: 10.')
parser.add_argument('--models_dir', type=str, nargs='?', action='store', default='Models/BBP_VGG11_models/theta_best.dat',
                    help='Where to save learnt weights and train vectors. Default: \'Models/BBP_VGG11_models/theta_best.dat\'.')
parser.add_argument('--results_dir', type=str, nargs='?', action='store', default='Results/BBP_VGG11_results/compare_sample.png',
                    help='Where to save learnt training plots. Default: \'Results/BBP_VGG11_results/compare_sample.png\'.')
parser.add_argument('--prior_sig', type=float, nargs='?', action='store', default=0.1,
                    help='Standard deviation of prior. Default: 0.1.')
parser.add_argument('--lr', type=float, nargs='?', action='store', default=1e-3,
                    help='learning rate. Default: 1e-3.')
args = parser.parse_args()



# ------------------------------------------------------------------------------------------------------
# train config
NTrainPointsCIFAR10 = 50000
batch_size = 32
log_interval = 1


# ------------------------------------------------------------------------------------------------------
# dataset
cprint('c', '\nData:')

# load data

# data augmentation
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,0.1307,0.1307), std=(0.3081,0.3801,0.3801))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,0.1307,0.1307), std=(0.3081,0.3801,0.3801))
])

use_cuda = torch.cuda.is_available()

trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
valset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

if use_cuda:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                              num_workers=3)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                            num_workers=3)

else:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=False,
                                              num_workers=3)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=False,
                                            num_workers=3)

## ---------------------------------------------------------------------------------------------------------------------
# net dims
cprint('c', '\nNetwork:')

nsamples = int(args.n_samples)  # How many samples to estimate ELBO with at each iteration
models_dir = args.models_dir
results_dir = args.results_dir
########################################################################################

#通常のnetの定義
net = BBP_Bayes_VGG11_Net(channels_in=3, side_in = 32, cuda=use_cuda, classes=10, batch_size=batch_size, Nbatches=(NTrainPointsCIFAR10 / batch_size), prior_instance=isotropic_gauss_prior(mu=0, sigma=args.prior_sig))
cprint('c', 'Reading %s\n' % models_dir)
state_dict = torch.load(models_dir)
net.epoch = state_dict['epoch']
net.lr = state_dict['lr']
for (param1, param2) in zip(net.model.parameters(),state_dict['model'].parameters()):
  param1.data = param2.data
net.optimizer = state_dict['optimizer']
print('  restoring epoch: %d, lr: %f' % (net.epoch, net.lr))
# net.load(models_dir)
net.set_mode_train(False)



# act dropを行うnetの定義
act_net = BBP_Bayes_VGG11_Net(channels_in=3, side_in = 32, cuda=use_cuda, classes=10, batch_size=batch_size, Nbatches=(NTrainPointsCIFAR10 / batch_size), prior_instance=isotropic_gauss_prior(mu=0, sigma=args.prior_sig),act_drop=True)
cprint('c', 'Reading %s\n' % models_dir)
state_dict = torch.load(models_dir)
act_net.epoch = state_dict['epoch']
act_net.lr = state_dict['lr']
for (param1, param2) in zip(act_net.model.parameters(),state_dict['model'].parameters()):
  param1.data = param2.data
act_net.optimizer = state_dict['optimizer']
print('  restoring epoch: %d, lr: %f' % (act_net.epoch, act_net.lr))
act_net.set_mode_train(False)







## ---------------------------------------------------------------------------------------------------------------------
# train
epoch = 0
cprint('c', '\nTest:')

print('  init cost variables:')

cost_dev = 0
err_dev = 0

# # We draw more samples on the first epoch in order to ensure convergence

err_results = np.zeros(nsamples)
ELBO_samples = nsamples

# for n in range(1,ELBO_samples+1):
#   cost_dev = 0
#   err_dev = 0
#   net.set_mode_train(False)
#   nb_samples = 0
#   for j, (x, y) in enumerate(valloader):
#     cost, err, probs = net.sample_eval(x, y, n)  # This takes the expected weights to save time, not proper inference
#     cost_dev += cost
#     err_dev += err
#     nb_samples += len(x)
#   cprint('g', '    Jdev = %f, err = %f\n' % (cost_dev.long()/nb_samples, err_dev.long()/nb_samples))
#   err_results[n-1] = err_dev.long()/nb_samples


# plotしたいとき
# plt.figure()
# plt.title('compare inference error different samples')
# plt.plot(np.arange(1,nsamples+1),err_results)
# plt.xlabel('number of samples')
# plt.ylabel('error')
# plt.savefig(results_dir)





# # 1サンプルだけしたいとき
# n = 1
# nb_samples = 0
# for j, (x, y) in enumerate(valloader):
#     cost, err, probs = net.sample_eval(x, y, n)  # This takes the expected weights to save time, not proper inference
#     cost_dev += cost
#     err_dev += err
#     nb_samples += len(x)
# cprint('g', '    Jdev = %f, err = %f\n' % (cost_dev.long()/nb_samples, err_dev.long()/nb_samples))



# # conv layerの描画を行う
# for i in range(8):
#   param_val = net.model.conv_out[i]
#   param_name = 'conv' + str(i) + '_out'
#   plt.figure()
#   plt.title(param_name)
#   tmp = param_val.cpu().detach().numpy()
#   val = tmp.flatten()
#   plt.hist(val,bins=100,color='deepskyblue')
#   plt.savefig('Graph' + '/Act/BBP_VGG11/' + param_name + '.png')



# # 通常convとact_drop convを比べる
# nb_samples = 0
# nb_equal = 0
# for j, (x, y) in enumerate(valloader):
#   prob_out1 = net.all_sample_eval(x, y, ELBO_samples)  # This takes the expected weights to save time, not proper inference
#   prob_out2 = act_net.all_sample_eval(x, y, ELBO_samples)
#   # 結果は(sample, batch, 10)となっている

#   # 予測結果を取り出すコード
#   pred1 = prob_out1[1:].mean(dim=0, keepdim=False).max(dim=1, keepdim=False)[1]
#   pred2 = prob_out2[1:].mean(dim=0, keepdim=False).max(dim=1, keepdim=False)[1]
#   nb_equal += torch.sum(pred1==pred2)
#   nb_samples += len(x)
# print('sample = ' + str(nb_samples))
# print('diff = ' + str(nb_samples - nb_equal))


# 確率分布の差の形で比べられるコード
for j, (x, y) in enumerate(valloader):
  prob_out1 = net.all_sample_eval(x, y, ELBO_samples)  # This takes the expected weights to save time, not proper inference
  prob_out2 = act_net.all_sample_eval(x, y, ELBO_samples)
  # 結果は(sample, batch, 10)となっている

  # 予測結果を取り出すコード
  pred1 = prob_out1[1:].max(dim=2, keepdim=False)[1]
  count1 = np.zeros((batch_size,10))
  for i in range(ELBO_samples-1):
    for j in range(batch_size):
      count1[j][pred1[i][j].data] += 1

  pred2 = prob_out2[1:].max(dim=2, keepdim=False)[1]
  count2 = np.zeros((batch_size,10))
  for i in range(ELBO_samples-1):
    for j in range(batch_size):
      count2[j][pred2[i][j].data] += 1

  for i in range(batch_size):
    print(count1[i])
    print(count2[i])
  print('end of batch')



