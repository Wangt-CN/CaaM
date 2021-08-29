import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR
import time
from sklearn.cluster import KMeans
from utils import get_network, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, update, get_mean_std, \
    Acc_Per_Context, Acc_Per_Context_Class, penalty, cal_acc, get_custom_network


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_env_ours(epoch, net, train_loader, args, training_opt, variance_opt, loss_function, optimizer, warmup_scheduler):
    start = time.time()
    env_num = variance_opt['n_env']
    assert isinstance(net, list)
    for net_ in net:
        net_.train()
    train_correct = 0.
    train_image_num = 0

    if variance_opt['erm_flag']:
        erm_dataloader_iterator = iter(train_loader[1])
    for batch_index, data_env in enumerate(train_loader[0]):
        if variance_opt['erm_flag']:
            try:
                data_env_erm = next(erm_dataloader_iterator)
            except StopIteration:
                erm_dataloader_iterator = iter(train_loader[1])
                data_env_erm = next(erm_dataloader_iterator)


        if epoch <= training_opt['warm']:
            for warmup_scheduler_ in warmup_scheduler:
                warmup_scheduler_.step()

        env_dic_nll = []
        env_dic_nll_spurious = []
        env_dic_acc = []
        images_all = []
        labels_all = []
        Erm_loss = torch.Tensor([0.])
        train_nll_spurious = torch.Tensor([0.])


        for edx, env in enumerate(data_env):
            images, labels, env_idx = env
            assert env_idx[0] == edx
            if args.gpu:
                labels = labels.cuda()
                images = images.cuda()
            images_all.append(images)
            labels_all.append(labels)
            causal_feature, spurious_feature, mix_feature = net[-1](images)
            causal_outputs = net[edx](causal_feature)
            batch_correct, train_acc = cal_acc(causal_outputs, labels)
            train_correct += batch_correct
            train_image_num += labels.size(0)
            env_dic_nll.append(loss_function(causal_outputs, labels))
            env_dic_acc.append(train_acc)

        train_nll = torch.stack(env_dic_nll).mean()
        train_acc = torch.stack(env_dic_acc).mean()

        ### 1. update feature extractor and classifier for irm
        ### ERM Loss
        loss = train_nll.clone()
        ### Variance Loss
        penalty_weight = float(variance_opt['penalty_weight']) if epoch >= variance_opt['penalty_anneal_iters'] else 1.0
        # penalty_weight = 0.0

        try:
            W_mean = torch.stack([net_.fc.weight for net_ in net[:env_num]], 0).mean(0)
            var_penalty = [(torch.norm(net_.fc.weight - W_mean, p=2) / torch.norm(net_.fc.weight, p=1)) ** 2 for net_ in net[:env_num]]
        except:
            W_mean = torch.stack([net_.module.fc.weight for net_ in net[:env_num]], 0).mean(0)
            var_penalty = [(torch.norm(net_.module.fc.weight - W_mean, p=2) / torch.norm(net_.module.fc.weight, p=1))**2 for net_ in net[:env_num]]
        loss_penalty = sum(var_penalty) / len(var_penalty)
        loss += penalty_weight * loss_penalty

        ### 2. update with erm loss
        if variance_opt['erm_flag']:
            images_erm, labels_erm = data_env_erm
            if args.gpu:
                labels_erm = labels_erm.cuda()
                images_erm = images_erm.cuda()
            if 'mixup' in training_opt and training_opt['mixup'] == True:
                inputs_erm, targets_a_erm, targets_b_erm, lam = mixup_data(images_erm, labels_erm, use_cuda=True)
                images_erm, targets_a_erm, targets_b_erm = map(Variable, (inputs_erm, targets_a_erm, targets_b_erm))

            _, __, mix_feature_erm = net[-1](images_erm)
            mix_outputs = net[-2](mix_feature_erm)

            if 'mixup' in training_opt and training_opt['mixup'] == True:
                Erm_loss = mixup_criterion(loss_function, mix_outputs, targets_a_erm, targets_b_erm, lam)
            else:
                Erm_loss = loss_function(mix_outputs, labels_erm)

            loss += Erm_loss

        for optimizer_ in optimizer:
            optimizer_.zero_grad()
        loss.backward()
        for optimizer_ in optimizer:
            optimizer_.step()

        ### 3. update feature extractor with spurious feature
        if variance_opt['sp_flag']:
            image_env = torch.cat(images_all, 0)
            _, spurious_feature_sp, __ = net[-1](image_env)
            # spurious_outputs = net[edx](spurious_feature_sp)
            try:
                W_mean = torch.stack([net_.fc.weight for net_ in net[:env_num]], 0).mean(0)
            except:
                W_mean = torch.stack([net_.module.fc.weight for net_ in net[:env_num]], 0).mean(0)
            spurious_outputs = torch.nn.functional.linear(spurious_feature_sp, W_mean)
            train_nll_spurious = smooth_loss(spurious_outputs, training_opt['classes'])
            optimizer[-1].zero_grad()
            train_nll_spurious.backward()
            optimizer[-1].step()


        if batch_index % training_opt['print_batch'] == 0:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tTrain_Loss: {:0.3f}\tNll_Loss: {:0.3f}\tPenalty: {:.2e}'
                  '\tPenalty_W: {:.1f}\tErm_Loss: {:.3f}\tSp_Loss: {:.3f}\tLR: {:0.6f}\tAcc: {:0.4f}'.format(
                loss.item(),
                train_nll.item(),
                loss_penalty.item(),
                penalty_weight,
                Erm_loss.item(),
                train_nll_spurious.item(),
                optimizer[0].param_groups[0]['lr'],
                train_acc,
                epoch=epoch,
                trained_samples=batch_index * training_opt['batch_size'] + len(images),
                total_samples=len(train_loader[0].dataset)
            ))
            print('\n')

    finish = time.time()
    # train_acc_all = train_correct / sum([len(env_dataloader.dataset) for env_dataloader in train_loader])
    train_acc_all = train_correct / train_image_num
    print('epoch {} training time consumed: {:.2f}s \t Train Acc: {:.4f}'.format(epoch, finish - start, train_acc_all))

    return train_acc_all


def auto_split(ref_model, pre_train_loader, pre_optimizer, pre_schedule, soft_split_all):
    if isinstance(ref_model, list):
        ref_model[-1].eval()
    else:
        ref_model.eval()
    low_loss = 1e5
    cnt = 0
    best_epoch = 0
    scale_update = True
    for epoch in range(100):
        pre_schedule.step()
        risk_all_list = []
        irm_risk_all_list = []
        reg_list = []
        for data in pre_train_loader:
            images, labels, idx = data
            labels = labels.cuda()
            images = images.cuda()
            if isinstance(ref_model, list):
                _, spurious_feature, __ = ref_model[0](images)
                outputs = ref_model[1](spurious_feature)
            else:
                outputs = ref_model(images)
            loss_value = F.cross_entropy(outputs, labels, reduction='none')
            # erm_risk = (soft_split * loss_value.unsqueeze(-1) / soft_split.sum(0)).sum(0)
            scale = torch.ones((1, outputs.size(-1))).cuda().requires_grad_()
            penalty = F.cross_entropy(outputs * scale, labels, reduction='none')

            split_logits = F.log_softmax(soft_split_all, dim=-1)
            hard_split_all = F.gumbel_softmax(split_logits, tau=1, hard=True)
            hard_split = hard_split_all[idx]
            penalty = (hard_split * penalty.unsqueeze(-1) / (hard_split.sum(0)+1e-20)).sum(0)
            erm_risk = (hard_split * loss_value.unsqueeze(-1) / (hard_split.sum(0)+1e-20)).sum(0)
            irm_risk_list = []
            for index in range(penalty.size(0)):
                irm_risk = torch.autograd.grad(penalty[index], [scale], create_graph=True)[0]
                irm_risk_list.append(torch.sum(irm_risk**2))
            irm_risk_final = - erm_risk.mean() - 1e6 * torch.stack(irm_risk_list).mean()
            if scale_update:
                scale_multi = irm_scale(irm_risk_final, -50)
                scale_update = False # every time just update once
            irm_risk_final *= scale_multi
            reg = torch.tensor([0.]).cuda()

            risk_all = irm_risk_final + reg
            risk_all_list.append(risk_all.item())
            irm_risk_all_list.append(irm_risk_final.item())
            reg_list.append(reg.item())
            pre_optimizer.zero_grad()
            risk_all.backward()
            pre_optimizer.step()

        avg_risk = sum(risk_all_list)/len(risk_all_list)
        avg_irm_risk = sum(irm_risk_all_list)/len(irm_risk_all_list)
        if epoch == 0:
            print("Initial_Irm_Risk: %.2f" % (avg_irm_risk))
        if avg_risk < low_loss:
            low_loss = avg_risk
            soft_split_best = soft_split_all.clone().detach()
            best_epoch = epoch
            cnt = 0
        else:
            cnt += 1
        print('\rInitializing Env [%d/%d]  Loss: %.2f  IRM_Risk: %.2f  Reg: %.2f  Cnt: %d  Lr: %.2f'
              %(epoch, 100, avg_risk, avg_irm_risk, sum(reg_list)/len(reg_list), cnt, pre_optimizer.param_groups[0]['lr']), end='',flush=True)

        if epoch > 80 and cnt > 5 or epoch == 99: #debug
            print('\nLoss not down. Break down training.  Epoch: %d  Loss: %.2f' %(best_epoch, low_loss))
            final_split_softmax = F.softmax(soft_split_best, dim=-1)
            print(final_split_softmax)
            return final_split_softmax, soft_split_best


def refine_split(bias_optimizer, bias_schedule, bias_classifier, classifier_train_loader, feature_extractor):
    ### optimize the bias classifier
    feature_extractor.eval()
    best_acc = 0.
    cnt = 0
    for epoch in range(100):
        bias_schedule.step()
        train_correct = 0
        for data in classifier_train_loader:
            images, labels = data
            labels = labels.cuda()
            images = images.cuda()
            with torch.no_grad():
                _, spurious_feature, __ = feature_extractor(images)
            bias_outputs = bias_classifier(spurious_feature)

            bias_loss = F.cross_entropy(bias_outputs, labels)
            bias_optimizer.zero_grad()
            bias_loss.backward()
            bias_optimizer.step()

            batch_correct, train_acc = cal_acc(bias_outputs, labels)
            train_correct += batch_correct
            print('\rOptimizing [%d/%d]  Loss: %.2f  Acc: %.2f  Lr: %.3f'  % (epoch, 100, bias_loss.item(), train_acc, bias_optimizer.param_groups[0]['lr']), end='', flush=True)

        train_acc_all = train_correct / len(classifier_train_loader.dataset)
        if train_acc_all > best_acc:
            best_acc = train_acc_all
            cnt = 0
        else:
            cnt += 1

        if cnt > 5 or epoch == 99: #debug
            print('\nAcc not up. Break down traning.  Epoch: %d  Acc: %.2f' %(epoch, best_acc))
            return bias_classifier, bias_optimizer


def auto_cluster(pre_train_loader, feature_extractor, training_opt, variance_opt):
    feature_extractor.eval()
    sp_feature_all = []

    with torch.no_grad():
        for data in pre_train_loader:
            images, labels, idx = data
            labels = labels.cuda()
            images = images.cuda()
            _, spurious_feature, __ = feature_extractor(images)
            sp_feature_all.append(spurious_feature)

    sp_feature_all = torch.cat(sp_feature_all, dim=0).cpu().numpy()

    km = KMeans(n_clusters=variance_opt['n_env'], random_state=training_opt['seed']).fit(sp_feature_all)
    cluster_split =  torch.zeros(km.labels_.size, variance_opt['n_env']).scatter_(1, torch.Tensor(km.labels_).long().unsqueeze(1), 1)

    return cluster_split


def smooth_loss(pred, classes):
    pred = F.log_softmax(pred, dim=-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(1 / classes)

    # loss = torch.mean(torch.sum(-true_dist * pred, dim=-1))
    loss = F.kl_div(pred, true_dist, reduction='batchmean')
    return loss


def update_pre_optimizer(soft_split):
    pre_optimizer = torch.optim.SGD([soft_split], lr=0.1, momentum=0.9, weight_decay=0)
    pre_optimizer.zero_grad()
    pre_optimizer.step()
    pre_scheduler = MultiStepLR(pre_optimizer, [40], gamma=0.2, last_epoch=-1)
    return pre_optimizer, pre_scheduler


def update_pre_optimizer_vit(soft_split):
    pre_optimizer = torch.optim.SGD([soft_split], lr=0.1, momentum=0.9, weight_decay=0)
    pre_optimizer.zero_grad()
    pre_optimizer.step()
    pre_scheduler = MultiStepLR(pre_optimizer, [30], gamma=0.1, last_epoch=-1)
    return pre_optimizer, pre_scheduler

def update_bias_optimizer(param):
    bias_optimizer = torch.optim.SGD(param, lr=0.01, momentum=0.9, weight_decay=0)
    bias_optimizer.zero_grad()
    bias_optimizer.step()
    bias_scheduler = MultiStepLR(bias_optimizer, [20], gamma=0.1, last_epoch=-1)
    return bias_optimizer, bias_scheduler


def irm_scale(irm_loss, default_scale=-100):
    with torch.no_grad():
        scale =  default_scale / irm_loss.clone().detach()
    return scale
