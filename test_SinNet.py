from __future__ import division, print_function
import time
import torch.utils.data
from torchvision import transforms, datasets
import argparse
import matplotlib
from src.Bayes_By_Backprop.test_SinNet import *

matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Test Bayesian Sin Net with Variational Inference')
parser.add_argument('n_samples', type=float, nargs='?', action='store', default=20,
                    help='How many MC samples to take when approximating the ELBO. Default: 20.')
parser.add_argument('--models_dir', type=str, nargs='?', action='store', default='BBP_SinNet_models/theta_best.dat',
                    help='Where to save learnt weights and train vectors. Default: \'BBP_SinNet_models/theta_best.dat\'.')
parser.add_argument('--results_dir', type=str, nargs='?', action='store', default='THESIS_DATA/5-2/',
                    help='Where to save learnt training plots. Default: \'THESIS_DATA/5-2/\'.')
parser.add_argument('--prior_sig', type=float, nargs='?', action='store', default=0.1,
                    help='Standard deviation of prior. Default: 0.1.')
parser.add_argument('--lr', type=float, nargs='?', action='store', default=1e-3,
                    help='learning rate. Default: 1e-3.')
args = parser.parse_args()


# Where to save models weights
models_dir = args.models_dir
# Where to save plots and error, accuracy vectors
results_dir = args.results_dir

# ------------------------------------------------------------------------------------------------------
# train config
batch_size = 20
log_interval = 1

# ------------------------------------------------------------------------------------------------------
# dataset
cprint('c', '\nData:')

# load data

use_cuda = torch.cuda.is_available()

## ---------------------------------------------------------------------------------------------------------------------
# net dims
cprint('c', '\nNetwork:')

lr = args.lr
nsamples = int(args.n_samples)  # How many samples to estimate ELBO with at each iteration
########################################################################################


net = BBP_Bayes_Sin_Net(lr=lr, channels_in=3, side_in = 32, cuda=use_cuda, classes=10, batch_size=batch_size, Nbatches=20, prior_instance=isotropic_gauss_prior(mu=0, sigma=args.prior_sig))
cprint('c', 'Reading %s\n' % models_dir)
state_dict = torch.load(models_dir)
net.epoch = state_dict['epoch']
net.lr = state_dict['lr']
for (param1, param2) in zip(net.model.parameters(),state_dict['model'].parameters()):
  param1.data = param2.data
net.optimizer = state_dict['optimizer']
print('  restoring epoch: %d, lr: %f' % (net.epoch, net.lr))
net.set_mode_train(False)

act_net = BBP_Bayes_Sin_Net(lr=lr, channels_in=3, side_in = 32, cuda=use_cuda, classes=10, batch_size=batch_size, Nbatches=20, prior_instance=isotropic_gauss_prior(mu=0, sigma=args.prior_sig), act_drop=True)
cprint('c', 'Reading %s\n' % models_dir)
state_dict = torch.load(models_dir)
act_net.epoch = state_dict['epoch']
act_net.lr = state_dict['lr']
for (param1, param2) in zip(act_net.model.parameters(),state_dict['model'].parameters()):
  param1.data = param2.data
act_net.optimizer = state_dict['optimizer']
print('  restoring epoch: %d, lr: %f' % (act_net.epoch, net.lr))
act_net.set_mode_train(False)


# trainの時に用いたデータ点の定義
train_x = torch.Tensor([-0.55,-0.4,-0.38,-0.35,-0.3,-0.25,-0.21,-0.15,-0.05,-0.03,0.0,0.05,0.06,0.17,0.30,0.35,0.4,0.5,0.9,0.95])
# train_x = torch.Tensor([-0.48,-0.38,-0.35,-0.33,-0.3,-0.25,-0.21,-0.15,-0.05,-0.03,0.0,0.05,0.06,0.20,0.30,0.35,0.41,0.5,0.96,0.99])
# train_x = torch.Tensor([-0.55,-0.4,-0.38,-0.35,-0.3,-0.25,-0.21,-0.15,-0.05,-0.03,0.0,0.05,0.06,0.20,0.30,0.35,0.4,0.5,0.96,0.99])

train_y = torch.sin(4*train_x) * torch.cos(14*train_x)

textsize = 15
marker = 5


# ## ---------------------------------------------------------------------------------------------------------------------
# # test normal network
# plt.figure()
# plt.xlabel('x')
# plt.ylabel('y')
# x = torch.arange(-1,1,0.01)
# y = torch.sin(4*x) * torch.cos(14*x)

# epoch = 0
# cprint('c', '\nTrain:')
# inference_results = np.ndarray((nsamples,len(x)))
# for n in range(nsamples):
#   result, _, _ = net.inference(x,y)
#   for i in range(len(x)):
#     inference_results[n][i] = result[i]

# for n in range(nsamples):
#   plt.plot(x, inference_results[n],linewidth=0.1)

# plt.plot(x,y)
# plt.scatter(train_x,train_y,color="blue")

# plt.savefig(results_dir+"5-2-normal.png")




# # activationのdropを行うコード
# act_net = BBP_Bayes_Sin_Net(lr=lr, channels_in=3, side_in = 32, cuda=use_cuda, classes=10, batch_size=batch_size, Nbatches=20, prior_instance=isotropic_gauss_prior(mu=0, sigma=args.prior_sig), act_drop=True)
# cprint('c', 'Reading %s\n' % models_dir)
# state_dict = torch.load(models_dir)
# act_net.epoch = state_dict['epoch']
# act_net.lr = state_dict['lr']
# for (param1, param2) in zip(act_net.model.parameters(),state_dict['model'].parameters()):
#   param1.data = param2.data
# act_net.optimizer = state_dict['optimizer']
# print('  restoring epoch: %d, lr: %f' % (act_net.epoch, net.lr))
# act_net.set_mode_train(False)

# plt.figure()
# plt.xlabel('x')
# plt.ylabel('y')
# x = torch.arange(-1,1,0.01)
# y = torch.sin(4*x) * torch.cos(14*x)


# alpha_list = [0.04,0.2,1]
# beta_list = [0.04,0.2,1]

# alpha = 1
# beta = 1
# epoch = 0
# cprint('c', '\nTrain:')
# inference_results = np.ndarray((nsamples,len(x)))
# sum_alpha = 0
# sum_beta = 0
# for n in range(nsamples):
#   if(n == 0):
#     result, drop_rate_alpha, drop_rate_beta = act_net.inference(x,y,first_sample=True)
#   else:
#     result, drop_rate_alpha, drop_rate_beta = act_net.inference(x,y,alpha=alpha,beta=beta)
#     sum_alpha = sum_alpha + drop_rate_alpha
#     sum_beta = sum_beta + drop_rate_beta
#   for i in range(len(x)):
#     inference_results[n][i] = result[i]

# print("alpha, beta = ", alpha, ", ", beta)
# print("drop rate alpha : ", sum_alpha/(nsamples-1))
# print("drop rate beta  : ", sum_beta/(nsamples-1))

# for n in range(nsamples):
#   plt.plot(x, inference_results[n],linewidth=0.1)

# plt.plot(x,y)
# plt.scatter(train_x,train_y,color="blue")

# plt.savefig(results_dir+"5-2-act_drop.png")



# ## ---------------------------------------------------------------------------------------------------------------------
# code for 5-2-compare
# ## ---------------------------------------------------------------------------------------------------------------------
plt.figure()
fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(20,10))
# normal net
x = torch.arange(-1,1,0.01)
y = torch.sin(4*x) * torch.cos(14*x)
epoch = 0
cprint('c', '\nTrain:')
inference_results = np.ndarray((nsamples,len(x)))
for n in range(nsamples):
  result, _, _ = net.inference(x,y)
  for i in range(len(x)):
    inference_results[n][i] = result[i]
# plot results
axes[0].set_ylim(-1.3,4.5)
for n in range(nsamples):
  axes[0].plot(x, inference_results[n],linewidth=0.1)
axes[0].set_title('no approximation')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].plot(x,y)
axes[0].scatter(train_x,train_y,color="blue")

# activation drop net
alpha = 0.005
beta = 0.2
epoch = 0
cprint('c', '\nTrain:')
inference_results = np.ndarray((nsamples,len(x)))
sum_alpha = 0
sum_beta = 0
for n in range(nsamples+1):
  if(n == 0):
    result, drop_rate_alpha, drop_rate_beta = act_net.inference(x,y,first_sample=True)
  else:
    result, drop_rate_alpha, drop_rate_beta = act_net.inference(x,y,alpha=alpha,beta=beta)
    sum_alpha = sum_alpha + drop_rate_alpha
    sum_beta = sum_beta + drop_rate_beta
  if (n!=0):
    for i in range(len(x)):
      inference_results[n-1][i] = result[i]

print("alpha, beta = ", alpha, ", ", beta)
print("drop rate alpha : ", sum_alpha/(nsamples))
print("drop rate beta  : ", sum_beta/(nsamples))
percent_alpha = str(int((sum_alpha/(nsamples))*1000)/1000)
percent_beta  = str(int((sum_beta /(nsamples))*1000)/1000)

axes[1].set_ylim(-1.3,4.5)
for n in range(nsamples):
  axes[1].plot(x, inference_results[n],linewidth=0.1)
axes[1].set_title("alpha = " + str(alpha)+ "(" + percent_alpha + "%)" + ", beta=" +str(beta)+ "(" + percent_beta + "%)")
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

axes[1].plot(x,y)
axes[1].scatter(train_x,train_y,color="blue")

plt.savefig(results_dir+"5-2-compare.png")



# ## ---------------------------------------------------------------------------------------------------------------------
# code for 5-2-parameters
# ## ---------------------------------------------------------------------------------------------------------------------


alpha_list = [0.005,0.03,0.09]
beta_list = [3,1,0.2]

plt.figure()
fig,axes = plt.subplots(nrows=len(alpha_list),ncols=len(beta_list),figsize=(20,16))
# normal net
x = torch.arange(-1,1,0.01)
y = torch.sin(4*x) * torch.cos(14*x)

for axes_i, beta in enumerate(beta_list):
  for axes_j, alpha in enumerate(alpha_list):
    ax = axes[axes_i,axes_j]
    epoch = 0
    cprint('c', '\nTrain:')
    inference_results = np.ndarray((nsamples,len(x)))
    sum_alpha = 0
    sum_beta = 0
    for n in range(nsamples+1):
      if(n == 0):
        result, drop_rate_alpha, drop_rate_beta = act_net.inference(x,y,first_sample=True)
      else:
        result, drop_rate_alpha, drop_rate_beta = act_net.inference(x,y,alpha=alpha,beta=beta)
        sum_alpha = sum_alpha + drop_rate_alpha
        sum_beta = sum_beta + drop_rate_beta
      if(n!=0):
        for i in range(len(x)):
          inference_results[n-1][i] = result[i]

    print("alpha, beta = ", alpha, ", ", beta)
    print("drop rate alpha : ", sum_alpha/(nsamples))
    print("drop rate beta  : ", sum_beta/(nsamples))
    percent_alpha = str(int((sum_alpha/(nsamples))*1000)/1000)
    percent_beta  = str(int((sum_beta /(nsamples))*1000)/1000)

    ax.set_title("alpha = " + str(alpha)+ "(" + percent_alpha + "%)" + ", beta=" +str(beta)+ "(" + percent_beta + "%)")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim(-1.3,4.5)
    for n in range(nsamples):
      ax.plot(x, inference_results[n],linewidth=0.1)

    ax.plot(x,y)
    ax.scatter(train_x,train_y,color="blue")


plt.savefig(results_dir+"5-2-parameteres.png")