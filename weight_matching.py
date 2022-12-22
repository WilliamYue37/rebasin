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

def compute_cost(cost_mat, perm):
    cost = 0
    for i in range(len(perm)):
        cost += cost_mat[i][perm[i]]

    return cost

def init_perms(params): # return identity undo_perms with num_of_layers length and identity perms with total layers length
    layers = list(params.keys())
    undo_perms, perms = [], []
    for i in range(len(layers) // 2):
        layer_name = layers[2 * i]

        length = params[layer_name].size()[0]
        perms.append(np.arange(length))

        length = params[layer_name].size()[1]
        undo_perms.append(np.arange(length))
        
    assert len(undo_perms) == len(layers) // 2 and len(perms) == len(layers) // 2
    return undo_perms, perms


def match_weights(a_params, b_params, iterations = 100):
    # find weight matching permutation
    layers = list(a_params.keys())
    num_of_layers = len(layers) // 2
    undo_perms, perms = init_perms(a_params)
    for i in range(iterations):
        progress = False
        for idx in torch.randperm(num_of_layers - 1): 
            # layer L weight (apply layer L undo perm)
            layer_name = layers[2 * idx]
            w_a = a_params[layer_name]
            w_b = b_params[layer_name]
            w_a = w_a.reshape((w_a.size(dim = 0), -1))
            w_b = w_b.reshape((w_b.size(dim = 0), -1))
            w_b = torch.index_select(w_b, 1, torch.from_numpy(undo_perms[idx])) # apply undo perm

            A = w_a @ w_b.T # compute cost matrix

            # layer L bias (no perm)
            layer_name = layers[2 * idx + 1]
            w_a = a_params[layer_name]
            w_b = b_params[layer_name]
            w_a = w_a.reshape((w_a.size(dim = 0), -1))
            w_b = w_b.reshape((w_b.size(dim = 0), -1))

            A += w_a @ w_b.T # compute cost matrix

            # layer L + 1 weight (apply layer L + 1 perm)
            layer_name = layers[2 * (idx + 1)]
            w_a = a_params[layer_name]
            w_b = b_params[layer_name]
            w_a = w_a.reshape((w_a.size(dim = 0), -1))
            w_b = w_b.reshape((w_b.size(dim = 0), -1))
            w_b = torch.index_select(w_b, 0, torch.from_numpy(perms[idx + 1])) # apply perm

            A += w_a.T @ w_b # compute cost matrix
            
            # compare with previous perm
            r_ind, c_ind = linear_sum_assignment(A.detach().numpy(), maximize = True)
            assert (torch.tensor(r_ind) == torch.arange(len(r_ind))).all()
            old_cost = compute_cost(A, perms[idx])
            new_cost = compute_cost(A, c_ind)
            progress = progress or new_cost > old_cost + 1e-12
                
            perms[idx] = undo_perms[idx + 1] = c_ind

        if not progress:
            break

    # apply perms
    new_params = copy.deepcopy(b_params)
    for idx in range(num_of_layers):
        # apply perm on weight layer
        layer_name = layers[2 * idx]
        w_b = b_params[layer_name]
        w_b = torch.index_select(w_b, 0, torch.from_numpy(perms[idx])) # rows
        w_b = torch.index_select(w_b, 1, torch.from_numpy(undo_perms[idx])) # cols
        new_params[layer_name] = w_b.squeeze()

        # apply perm on bias layer
        layer_name = layers[2 * idx + 1]
        w_b = b_params[layer_name]
        w_b = torch.index_select(w_b, 0, torch.from_numpy(perms[idx])) # rows
        new_params[layer_name] = w_b.squeeze()

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
