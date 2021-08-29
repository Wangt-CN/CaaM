import torch
import torch.nn as nn
import os
import json
import tabulate
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


@torch.no_grad()
def eval_best(config, args, net, test_loader, loss_function ,checkpoint_path, best_epoch):
    start = time.time()
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

    test_loss = 0.0 # cost function error
    correct = 0.0
    label2train = test_loader.dataset.label2train
    label2train = {v: k for k, v in label2train.items()}
    acc_per_context = Acc_Per_Context_Class(config['cxt_dic_path'], list(label2train.keys()))

    for (images, labels, context) in test_loader:

        if args.gpu:
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
    print('Evaluate Acc Per Context Per Class...')
    class_dic = json.load(open(config['class_dic_path'], 'r'))
    class_dic = {v: k for k, v in class_dic.items()}
    acc_cxt_all_class = acc_per_context.cal_acc()
    for label_class in acc_cxt_all_class.keys():
        acc_class = acc_cxt_all_class[label_class]
        print('Class: %s' %(class_dic[int(label2train[label_class])]))
        print(tabulate.tabulate(acc_class, headers=['Context', 'Acc'], tablefmt='grid'))

    return correct.float() / len(test_loader.dataset)


@torch.no_grad()
def eval_mode(config, args, net, test_loader, loss_function, model_path):
    start = time.time()
    load_model(net, model_path)
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    label2train = test_loader.dataset.label2train
    label2train = {v: k for k, v in label2train.items()}
    acc_per_context = Acc_Per_Context_Class(config['cxt_dic_path'], list(label2train.keys()))

    for (images, labels, context) in test_loader:

        if args.gpu:
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

    return correct.float() / len(test_loader.dataset)