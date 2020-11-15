import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='draw weight mean and diviation')

parser.add_argument('model_dir',type=str,help='specify the place of directory')
parser.add_argument('graph_dir',type=str,help='specify the place for graph')

args = parser.parse_args()

model_dir = args.model_dir
graph_dir = args.graph_dir

model = torch.load(model_dir)

print(model)

# for name in model['model'].state_dict().keys():
#   param_name = name
#   param_val = model['model'].parameters(){name}

for i in range(len(model['model'].state_dict().keys())):
  param_name = list(model['model'].state_dict().keys())[i]
  param_val = list(model['model'].parameters())[i]

  if i % 2 == 1:
    param_val = 1e-6 + torch.nn.functional.softplus(param_val, beta=1, threshold=20)

  plt.figure()
  plt.title(param_name)
  # plt.xlim(-0.2,0.2)
  tmp = param_val.cpu().detach().numpy()
  val = tmp.flatten()
  plt.hist(val,bins=100,color='deepskyblue')
  plt.savefig('Graph' + '/' + graph_dir + '/' + param_name + '.png')



