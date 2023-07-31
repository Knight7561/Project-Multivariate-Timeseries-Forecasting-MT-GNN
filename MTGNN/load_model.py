import argparse
import json
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch.nn as nn

from model import MTGNN_Model
from trainer import Optim
from util import *


def evaluate(data, X, Y, model, eval_mse_loss, eval_l1_loss, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predicted_label = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        if len(output.shape) == 1:
            output = output.unsqueeze(dim=0)
        if predicted_label is None:
            predicted_label = output
            test = Y
        else:
            predicted_label = torch.cat((predicted_label, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        total_loss += eval_mse_loss(output * scale, Y * scale).item()
        total_loss_l1 += eval_l1_loss(output * scale, Y * scale).item()
        n_samples += (output.size(0) * data.m)

    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predicted_label = predicted_label.data.cpu().numpy()
    actual_label = test.data.cpu().numpy()
    sigma_p = (predicted_label).std(axis=0)
    sigma_g = (actual_label).std(axis=0)
    mean_p = predicted_label.mean(axis=0)
    mean_g = actual_label.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predicted_label - mean_p) * (actual_label - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return rse, rae, correlation, predicted_label, actual_label


def train(data, X, Y, model, loss_function, optimizer, batch_size):
    model.train()
    total_loss = []
    n_samples = 0
    batch = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        if batch % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]
            id = torch.tensor(id).to(device)
            tx = X[:, :, id, :]
            ty = Y[:, id]
            output = model(tx, id)
            output = torch.squeeze(output)
            scale = data.scale.expand(output.size(0), data.m)
            scale = scale[:, id]
            loss = loss_function(output * scale, ty * scale)
            loss.backward()
            total_loss.append(loss.item() / (output.size(0) * data.m))
            n_samples += (output.size(0) * data.m)
            optimizer.step()

        if batch % 100 == 0:
            loss = loss.item() / (output.size(0) * data.m)
            print(f"Batch:{batch} | loss: {loss}")
        batch += 1
    average_loss = sum(total_loss) / n_samples
    return average_loss, total_loss


parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='./data/solar.txt',
                    help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--output_path', type=str, default='results',
                    help='path to save the final results like images, json, numpy files')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device', type=str, default='cuda:1', help='')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
parser.add_argument('--num_nodes', type=int, default=137, help='number of nodes/variables')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--subgraph_size', type=int, default=20, help='k')
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
parser.add_argument('--dilation_exponential', type=int, default=2, help='dilation exponential')
parser.add_argument('--conv_channels', type=int, default=16, help='convolution channels')
parser.add_argument('--residual_channels', type=int, default=16, help='residual channels')
parser.add_argument('--skip_channels', type=int, default=32, help='skip channels')
parser.add_argument('--end_channels', type=int, default=64, help='end channels')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--seq_in_len', type=int, default=24 * 7, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=1, help='output sequence length')
parser.add_argument('--horizon', type=int, default=3)
parser.add_argument('--layers', type=int, default=5, help='number of layers')

parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay rate')

parser.add_argument('--clip', type=int, default=5, help='clip')

parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
parser.add_argument('--tanhalpha', type=float, default=3, help='tanh alpha')

parser.add_argument('--epochs', type=int, default=1, help='')
parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
parser.add_argument('--step_size', type=int, default=100, help='step_size')
parser.add_argument('--no_of_runs', type=int, default=1, help='No of times model has to run')
parser.add_argument('--sample_data', type=int, default=0, help='Sample no of records to run for benchmark')

args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)


def main():
    Path(args.output_path).mkdir(exist_ok=True)
    data = DataLoader(args.data, 0.6, 0.2, device, args.horizon, args.seq_in_len, args.sample_data, args.normalize)

    model = MTGNN_Model(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
                        device, dropout=args.dropout, subgraph_size=args.subgraph_size,
                        node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
                        conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                        skip_channels=args.skip_channels, end_channels=args.end_channels,
                        seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                        layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha,
                        layer_norm_affline=False)
    model = model.to(device)

    print('The receptive field size is', model.receptive_field)
    model_parameters = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', model_parameters, flush=True)

    if args.L1Loss:
        train_loss_func = nn.L1Loss(size_average=False).to(device)
    else:
        train_loss_func = nn.MSELoss(size_average=False).to(device)
    eval_mse_loss = nn.MSELoss(size_average=False).to(device)
    eval_l1_loss = nn.L1Loss(size_average=False).to(device)

    best_val = 10000000
    optimizer = Optim(model.parameters(), args.optim, args.lr, args.clip, lr_decay=args.weight_decay)
    all_batches_loss = []

    # Load the best saved model.
    with open(args.output_path+"/"+args.save, 'rb') as f:
        model = torch.load(f)

    vtest_acc, vtest_rae, vtest_corr, _, _ = evaluate(data, data.valid[0], data.valid[1], model, eval_mse_loss,
                                                      eval_l1_loss, args.batch_size)
    test_acc, test_rae, test_corr, predicted_output, actual_output = evaluate(data, data.test[0], data.test[1], model,
                                                                              eval_mse_loss, eval_l1_loss,
                                                                              args.batch_size)
    print(f"Final Test RSE {test_acc} | Test RAE {test_rae} | Test Correlation {test_corr}")

    ## Save all the results:
    adjacency_matrix = model.gc(torch.arange(args.num_nodes).to(device))
    adjacency_matrix = adjacency_matrix.data.cpu().numpy()
    result_dict = {"training_loss": all_batches_loss}
    np.save(f"{args.output_path}/adjacency_matrix", adjacency_matrix)
    np.save(f"{args.output_path}/predicted_output", predicted_output)
    np.save(f"{args.output_path}/actual_output", actual_output)
    with open(F"{args.output_path}/result.json", "w") as outfile:
        json.dump(result_dict, outfile)
    plt.plot(range(len(all_batches_loss)), all_batches_loss)
    plt.savefig(f'{args.output_path}/train_loss.png')

    return vtest_acc, vtest_rae, vtest_corr, test_acc, test_rae, test_corr


if __name__ == "__main__":
    valid_accuracy, valid_rae, valid_correlation = [], [], []
    test_accuracy, test_RAE, test_correlation = [], [], []
    no_of_runs = args.no_of_runs
    for i in range(no_of_runs):
        val_acc, val_rae, val_corr, test_acc, test_rae, test_corr = main()
        valid_accuracy.append(val_acc)
        valid_rae.append(val_rae)
        valid_correlation.append(val_corr)
        test_accuracy.append(test_acc)
        test_RAE.append(test_rae)
        test_correlation.append(test_corr)
    print(f'Average of {no_of_runs} runs')
    print(
        f"Valid RSE Mean: {np.mean(valid_accuracy)}, Valid RAE Mean: {np.mean(valid_rae)}, Valid Correlation Mean: {np.mean(valid_correlation)}")
    print(f"Valid RSE Standard Deviation: {np.std(valid_accuracy)}, Valid RAE Standard Deviation: {np.std(valid_rae)}, "
          f"Valid Correlation Standard Deviation: {np.std(valid_correlation)}")
    print(
        f"Test RSE Mean: {np.mean(test_accuracy)}, Test RAE Mean: {np.mean(test_RAE)}, Test Correlation Mean: {np.mean(test_correlation)}")
    print(f"Test RSE Standard Deviation: {np.std(test_accuracy)}, Test RAE Standard Deviation: {np.std(test_RAE)}, "
          f"Test Correlation Standard Deviation: {np.std(test_correlation)}")
