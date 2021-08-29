import torch
import torch.nn as nn
import os
import json
import tabulate
import numpy as np
import random
import time
from utils import Acc_Per_Context, Acc_Per_Context_Class, cal_acc, save_model, load_model


@torch.no_grad()
def eval_training(config, args, net, test_loader, loss_function, writer, epoch=0, tb=True):
    start = time.time()
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    acc_per_context = Acc_Per_Context(config['cxt_dic_path'])

    for (images, labels, context) in test_loader:

        images = images.cuda()
        labels = labels.cuda()

        if isinstance(net, list):
            feature = net[-1](images)
            if isinstance(feature, list):
                feature = feature[0]  # chosse causal feature
            try:
                W_mean = torch.stack([net_.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            except:
                W_mean = torch.stack([net_.module.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            outputs = nn.functional.linear(feature, W_mean)
        else:
            outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        acc_per_context.update(preds, labels, context)

    finish = time.time()
    print('Evaluating Network.....')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset),
        finish - start
    ))
    print('Evaluate Acc Per Context...')
    acc_cxt = acc_per_context.cal_acc()
    print(tabulate.tabulate(acc_cxt, headers=['Context', 'Acc'], tablefmt='grid'))

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)

    return correct.float() / len(test_loader.dataset)


def eval_training_imagenet(config, args, net, test_loader, loss_function, writer, epoch=0, tb=True):
    start = time.time()
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels, context) in test_loader:

        images = images.cuda()
        labels = labels.cuda()

        if isinstance(net, list):
            feature = net[-1](images)
            if isinstance(feature, list):
                feature = feature[0]  # chosse causal feature
            try:
                W_mean = torch.stack([net_.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            except:
                W_mean = torch.stack([net_.module.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            outputs = nn.functional.linear(feature, W_mean)
        else:
            outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    print('Evaluating Network.....')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset),
        finish - start
    ))

    return correct.float() / len(test_loader.dataset)


@torch.no_grad()
def evaluate_rebias(dataloader, net, config,
                    num_classes=9,
                    num_clusters=9,
                    num_cluster_repeat=3,
                    key=None):
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()
    total = 0
    f_correct = 0
    num_correct = [np.zeros([num_classes, num_clusters]) for _ in range(num_cluster_repeat)]
    num_instance = [np.zeros([num_classes, num_clusters]) for _ in range(num_cluster_repeat)]

    for images, labels, bias_labels in dataloader:
        images = images.to('cuda')
        labels = labels.to('cuda')
        for bias_label in bias_labels:
            bias_label.to('cuda')

        if isinstance(net, list):
            feature = net[-1](images)
            if isinstance(feature, list):
                feature = feature[0]  # chosse causal feature
            try:
                W_mean = torch.stack([net_.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            except:
                W_mean = torch.stack([net_.module.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            outputs = nn.functional.linear(feature, W_mean)
        else:
            outputs = net(images)

        batch_size = labels.size(0)
        total += batch_size

        if key == 'unbiased':
            num_correct, num_instance = imagenet_unbiased_accuracy(outputs.data, labels, bias_labels,
                                                                        num_correct, num_instance, num_cluster_repeat)
        else:
            f_correct += n_correct(outputs, labels)


    if key == 'unbiased':
        for k in range(num_cluster_repeat):
            x, y = [], []
            _num_correct, _num_instance = num_correct[k].flatten(), num_instance[k].flatten()
            for i in range(_num_correct.shape[0]):
                __num_correct, __num_instance = _num_correct[i], _num_instance[i]
                if __num_instance >= 10:
                    x.append(__num_instance)
                    y.append(__num_correct / __num_instance)
            f_correct += sum(y) / len(x)

        ret = {'f_acc': f_correct / num_cluster_repeat}
    else:
        ret = {'f_acc': f_correct / total}

    return ret


@torch.no_grad()
def eval_best(config, args, net, test_loader, loss_function ,checkpoint_path, best_epoch):

    try:
        load_model(net, checkpoint_path.format(net=args.net, epoch=best_epoch, type='best'))
        # net.load_state_dict(torch.load(checkpoint_path.format(net=args.net, epoch=best_epoch, type='best')))
    except:
        print('no best checkpoint')
        load_model(net, checkpoint_path.format(net=args.net, epoch=180, type='regular'))
        # net.load_state_dict(torch.load(checkpoint_path.format(net=args.net, epoch=180, type='regular')))
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()

    scores = {}
    for key, val_loader in test_loader.items():
        scores[key] = evaluate_rebias(val_loader, net, config, num_classes=9, key=key)

    return scores


@torch.no_grad()
def eval_mode(config, net, test_loader, model_path):

    load_model(net, model_path)
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()

    scores = {}
    for key, val_loader in test_loader.items():
        if key != 'imagenet-a':
            scores[key] = evaluate_rebias(val_loader, net, config, num_classes=9, key=key)

    return scores


def n_correct(pred, labels):
    _, predicted = torch.max(pred.data, 1)
    n_correct = (predicted == labels).sum().item()
    return n_correct


def imagenet_unbiased_accuracy(outputs, labels, cluster_labels,
                               num_correct, num_instance,
                               num_cluster_repeat=3):
    for j in range(num_cluster_repeat):
        for i in range(outputs.size(0)):
            output = outputs[i]
            label = labels[i]
            cluster_label = cluster_labels[j][i]

            _, pred = output.topk(1, 0, largest=True, sorted=True)
            correct = pred.eq(label).view(-1).float()

            num_correct[j][label][cluster_label] += correct.item()
            num_instance[j][label][cluster_label] += 1

    return num_correct, num_instance
