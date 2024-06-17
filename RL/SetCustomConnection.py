import ncps.wirings
import numpy
import numpy as np
import torch
import torch.nn as nn
from cfc import *


def SetCustomConnectionFromText(CFCNetwork: CfC, path):
    with open(path, 'r') as file:
        content = file.readlines()

    layer_index = -1
    row_index = -1

    sparsity_masks = []
    modify_masks = []

    spa_temp = []
    mod_temp = []

    for line in content:
        if "si" in line:
            layer_index = 0
            row_index = 0
            continue
        elif "ic" in line:
            layer_index = 1
            row_index = 0
            sparsity_masks.append(torch.stack(spa_temp))
            modify_masks.append(torch.stack(mod_temp))
            spa_temp = []
            mod_temp = []
            continue
        elif "co" in line:
            layer_index = 2
            row_index = 0
            sparsity_masks.append(torch.stack(spa_temp))
            modify_masks.append(torch.stack(mod_temp))
            spa_temp = []
            mod_temp = []
            continue
        elif len(line) < 2:
            continue
        else:
            elements = line.split('\t')
            stmp = torch.zeros(len(elements))
            mtmp = torch.zeros(len(elements))

            for i, e in enumerate(elements):
                if '1' in e:
                    stmp[i] = 1
                    mtmp[i] = 1
                elif '0' in e:
                    mtmp[i] = 1

            spa_temp.append(stmp)
            mod_temp.append(mtmp)

    sparsity_masks.append(torch.stack(spa_temp))
    modify_masks.append(torch.stack(mod_temp))

    # print(sparsity_masks)
    # print(modify_masks)

    SetCustomConnection(CFCNetwork, sparsity_masks, modify_masks)


def SetCustomConnection(CFCNetwork: CfC, sparsity_masks, modify_masks):
    assert len(sparsity_masks) == 3

    state_dict = CFCNetwork.state_dict()

    for i in range(len(sparsity_masks)):
        mask = sparsity_masks[i]
        modify_mask = modify_masks[i]
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        mask.requires_grad = False

        if isinstance(modify_mask, np.ndarray):
            modify_mask = torch.from_numpy(modify_mask)

        modify_mask.requires_grad = False

        state_dict[f"rnn_cell.layer_{i}.sparsity_mask"].data = state_dict[f"rnn_cell.layer_{i}.sparsity_mask"].data * (
                    1 - modify_mask) + mask * modify_mask

    CFCNetwork.load_state_dict(state_dict)

def MatchWiring(CFCNetwork: CfC):
    wiring = CFCNetwork.wiring
    wiredCfC = CFCNetwork.rnn_cell

    for i, layer in enumerate(wiredCfC._layers):
        hidden_units = wiring.get_neurons_of_layer(i)
        if i == 0:
            sensory_count = wiring.sensory_adjacency_matrix.shape[0]
            wiring.sensory_adjacency_matrix[:, hidden_units] = layer.sparsity_mask.detach().cpu().numpy().T[:sensory_count]
            for j, unit in enumerate(hidden_units):
                wiring.adjacency_matrix[unit, hidden_units] = layer.sparsity_mask.detach().cpu().numpy().T[sensory_count + j, :]
        else:
            last_units = wiring.get_neurons_of_layer(i - 1)
            for j, unit in enumerate(last_units + hidden_units):
                wiring.adjacency_matrix[unit, hidden_units] = layer.sparsity_mask.detach().cpu().numpy().T[j, :]


if __name__ == "__main__":
    SetCustomConnectionFromText(None, "./sparsity.txt")
