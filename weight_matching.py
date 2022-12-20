import argparse
from mlp import MLP
import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
from tqdm import tqdm
import copy
from train import train, test
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from plot import plot_interp_acc

def match_weights(a_params, b_params):
    # find weight matching permutation
    new_params = copy.deepcopy(b_params)
    layers = list(a_params.keys())
    n = len(layers)
    undo_perms = [None] * n
    for idx in torch.randperm(n // 2 - 1): # only care about weight layers and don't care about last layer
        layer_name = layers[2 * idx]
        w_a = a_params[layer_name]
        w_b = b_params[layer_name]
        w_a = w_a.reshape((w_a.size(dim = 0), -1))
        w_b = w_b.reshape((w_b.size(dim = 0), -1))

        A = w_a @ w_b.T

        r_ind, c_ind = linear_sum_assignment(A.detach().numpy(), maximize = True)
            
        undo_perms[idx] = c_ind
        new_w_b = torch.index_select(w_b, 0, torch.from_numpy(c_ind))
        new_params[layer_name] = new_w_b.squeeze()

        # update bias layer with same perm
        bias_layer_name = layers[2 * idx + 1]
        bias_layer = b_params[bias_layer_name]
        new_bias_layer = torch.index_select(bias_layer, 0, torch.from_numpy(c_ind))
        new_params[bias_layer_name] = new_bias_layer.squeeze()

    # apply undo perms
    for idx in range(1, n // 2):
        layer_name = layers[2 * idx]
        w_b = new_params[layer_name]
        new_w_b = torch.index_select(w_b, 1, torch.from_numpy(undo_perms[idx - 1]))
        new_params[layer_name] = new_w_b.squeeze()

    return new_params
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", type=str, required=True)
    parser.add_argument("--model_b", type=str, required=True)
    args = parser.parse_args()

    model_a = MLP()
    model_b = MLP()
    checkpoint = torch.load(args.model_a)
    model_a.load_state_dict(checkpoint)
    checkpoint_b = torch.load(args.model_b)
    model_b.load_state_dict(checkpoint_b)

    #test_a = {'layer0.weight': torch.tensor([[3, 5], [1, 2]]), 'layer0.bias': torch.tensor([0, 0]), 'layer1.weight': torch.tensor([[3, 5], [1, 2]]), 'layer1.bias': torch.tensor([0, 0]), 'layer2.weight': torch.tensor([[0, 0], [0, 0]]), 'layer2.bias': torch.tensor([0, 0])}
    #test_b = {'layer0.weight': torch.tensor([[1, 2], [3, 4]]), 'layer0.bias': torch.tensor([1, 2]), 'layer1.weight': torch.tensor([[1, 2], [3, 4]]), 'layer1.bias': torch.tensor([1, 2]), 'layer2.weight': torch.tensor([[1, 2], [3, 4]]), 'layer2.bias': torch.tensor([1, 2])}

    #test_params = match_weights(test_a, test_b)

    new_params = match_weights(model_a.state_dict(), model_b.state_dict())
    
    # test against mnist
    transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
      ])
    test_kwargs = {'batch_size': 5000}
    train_kwargs = {'batch_size': 5000}
    dataset = datasets.MNIST('../data', train=False,
                      transform=transform)
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                      transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)
    lambdas = torch.linspace(0, 1, steps=25)

    # merge models
    test_acc, train_acc = [], []
    model_b.load_state_dict(new_params)
    model_a_dict = copy.deepcopy(model_a.state_dict())
    model_b_dict = copy.deepcopy(model_b.state_dict())
    for lam in tqdm(lambdas):
        merged_params = copy.deepcopy(model_a_dict)
        for p in model_a_dict:
            merged_params[p] = (1 - lam) * model_b_dict[p] + lam * model_a_dict[p]
        
        model_b.load_state_dict(merged_params)
        test_loss, acc = test(model_b, 'cpu', test_loader)
        test_acc.append(acc)
        train_loss, acc = test(model_b, 'cpu', train_loader)
        train_acc.append(acc)

    fig = plot_interp_acc(lambdas, train_acc, test_acc)
    plt.savefig(f"mnist_mlp_weight_matching_interp_accuracy_epoch.png", dpi=300)

if __name__ == "__main__":
    main()
