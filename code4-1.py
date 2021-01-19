import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from src.Bayes_By_Backprop.act_test_VGG11batchnorm import *


# ------------------------------------------------------------------------------------------------------
# config
NTrainPointsCIFAR10 = 50000
batch_size = 32
log_interval = 1
nsamples = 100


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

########################################################################################

## ---------------------------------------------------------------------------------------------------------------------
# code for evaluation 4-1
## ---------------------------------------------------------------------------------------------------------------------
# まず，分散などを描画するコード
parser = argparse.ArgumentParser(description='draw weight mean and diviation')

parser.add_argument('model_dir',type=str,help='specify the place of directory')

args = parser.parse_args()

model_dir = args.model_dir

model = torch.load(model_dir)
print(model)

count = 0
count2 = 0
for i in range(len(model['model'].state_dict().keys())):
  param_name = list(model['model'].state_dict().keys())[i]
  if ("running_mean" in param_name or "running_var" in param_name or "num_batches_tracked" in param_name):
    continue
  param_val = list(model['model'].parameters())[count]
  count = count + 1
  if "batchnorm" in param_name:
    continue

  if count2 % 2 == 1:
    param_val = 1e-6 + torch.nn.functional.softplus(param_val, beta=1, threshold=20)

  plt.figure()
  if (count2 < 32):
    graph_title = 'conv' + str(int(count2/4) + 1)
  else :
    graph_title = 'fc' + str(int((count2-32)/4) + 1)
  if (count2 % 4 == 0):
    graph_title = graph_title + " weight mean"
  elif (count2 % 4 == 1):
    graph_title = graph_title + " weight standard deviation"
  elif (count2 % 4 == 2):
    graph_title = graph_title + " bias mean"
  else :
    graph_title = graph_title + " bias standard deviation"
  print(param_name)
  print(graph_title)
  # plt.title(graph_title.capitalize())
  plt.xlabel('Value')
  plt.ylabel('Count')
  # plt.xlim(-0.2,0.2)
  tmp = param_val.cpu().detach().numpy()
  val = tmp.flatten()
  plt.hist(val,bins=100,color='deepskyblue')
  plt.savefig('THESIS_DATA/4-1'+ '/distribution/' + param_name + '.png')
  count2 = count2 + 1





# ここからは，input - input_firstの値を調べるコード
# act dropを行うnetの定義
act_net = BBP_Bayes_VGG11_Net(channels_in=3, side_in = 32, cuda=use_cuda, classes=10, batch_size=batch_size, Nbatches=(NTrainPointsCIFAR10 / batch_size), prior_instance=isotropic_gauss_prior(mu=0, sigma=0.1),act_drop=True)
cprint('c', 'Reading %s\n' % model_dir)
state_dict = torch.load(model_dir)
act_net.epoch = state_dict['epoch']
act_net.lr = state_dict['lr']
for (param1, param2) in zip(act_net.model.parameters(),state_dict['model'].parameters()):
  param1.data = param2.data
act_net.optimizer = state_dict['optimizer']
print('  restoring epoch: %d, lr: %f' % (act_net.epoch, act_net.lr))
act_net.set_mode_train(False)

ELBO_samples = nsamples
for batch, (x, y) in enumerate(valloader):
  # 結果は(sample, batch, 10)となっている
  prob_out2, drop_rate_alpha, drop_rate_beta = act_net.all_sample_eval(x, y, ELBO_samples, alpha=0, beta=0)

  # plt.figure()
  # tmp = act_net.model.conv1.diff.cpu().detach().numpy()
  # val = tmp.flatten()
  # plt.hist(val,bins=100,color='deepskyblue')
  # plt.savefig('THESIS_DATA/4-1/diff/'+ 'conv1.png')
  plt.figure()
  tmp = act_net.model.conv2.diff.cpu().detach().numpy()
  val = tmp.flatten()
  plt.hist(val,bins=100,color='deepskyblue')
  # plt.title('Input difference in conv2')
  plt.xlabel('Value')
  plt.ylabel('Count')
  plt.savefig('THESIS_DATA/4-1/diff/'+ 'conv2.png')
  plt.figure()
  tmp = act_net.model.conv3.diff.cpu().detach().numpy()
  val = tmp.flatten()
  plt.hist(val,bins=100,color='deepskyblue')
  # plt.title('Input difference in conv3')
  plt.xlabel('Value')
  plt.ylabel('Count')
  plt.savefig('THESIS_DATA/4-1/diff/'+ 'conv3.png')
  plt.figure()
  tmp = act_net.model.conv4.diff.cpu().detach().numpy()
  val = tmp.flatten()
  plt.hist(val,bins=100,color='deepskyblue')
  # plt.title('Input difference in conv4')
  plt.xlabel('Value')
  plt.ylabel('Count')
  plt.savefig('THESIS_DATA/4-1/diff/'+ 'conv4.png')
  plt.figure()
  tmp = act_net.model.conv5.diff.cpu().detach().numpy()
  val = tmp.flatten()
  plt.hist(val,bins=100,color='deepskyblue')
  # plt.title('Input difference in conv5')
  plt.xlabel('Value')
  plt.ylabel('Count')
  plt.savefig('THESIS_DATA/4-1/diff/'+ 'conv5.png')
  plt.figure()
  tmp = act_net.model.conv6.diff.cpu().detach().numpy()
  val = tmp.flatten()
  plt.hist(val,bins=100,color='deepskyblue')
  # plt.title('Input difference in conv6')
  plt.xlabel('Value')
  plt.ylabel('Count')
  plt.savefig('THESIS_DATA/4-1/diff/'+ 'conv6.png')
  plt.figure()
  tmp = act_net.model.conv7.diff.cpu().detach().numpy()
  val = tmp.flatten()
  plt.hist(val,bins=100,color='deepskyblue')
  # plt.title('Input difference in conv7')
  plt.xlabel('Value')
  plt.ylabel('Count')
  plt.savefig('THESIS_DATA/4-1/diff/'+ 'conv7.png')
  plt.figure()
  tmp = act_net.model.conv8.diff.cpu().detach().numpy()
  val = tmp.flatten()
  plt.hist(val,bins=100,color='deepskyblue')
  # plt.title('Input difference in conv8')
  plt.xlabel('Value')
  plt.ylabel('Count')
  plt.savefig('THESIS_DATA/4-1/diff/'+ 'conv8.png')
  plt.figure()
  tmp = act_net.model.bfc1.diff.cpu().detach().numpy()
  val = tmp.flatten()
  plt.hist(val,bins=100,color='deepskyblue')
  plt.savefig('THESIS_DATA/4-1/diff/'+ 'bfc1.png')
  # plt.title('Input difference in fc1')
  plt.xlabel('Value')
  plt.ylabel('Count')
  plt.figure()
  tmp = act_net.model.bfc2.diff.cpu().detach().numpy()
  val = tmp.flatten()
  plt.hist(val,bins=100,color='deepskyblue')
  # plt.title('Input difference in fc2')
  plt.xlabel('Value')
  plt.ylabel('Count')
  plt.savefig('THESIS_DATA/4-1/diff/'+ 'bfc2.png')
  plt.figure()
  tmp = act_net.model.bfc3.diff.cpu().detach().numpy()
  val = tmp.flatten()
  plt.hist(val,bins=100,color='deepskyblue')
  # plt.title('Input difference in fc3')
  plt.xlabel('Value')
  plt.ylabel('Count')
  plt.savefig('THESIS_DATA/4-1/diff/'+ 'bfc3.png')



  break