import argparse
import numpy as np
import math
import torch.autograd as autograd
import helper
from mlp import Recognizer_mlp
from mlp_cifar import Recognizer_mlp_cifar
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from sklearn.metrics import auc, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--max_epochs', type=int, default=50,
                    help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=100,
                    help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='adam: learning rate')
parser.add_argument('--n_cpu', type=int, default=12,
                    help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=10,
                    help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=784,
                    help='size the image')
parser.add_argument('--num_classes', type=int, default=10,
                    help='numbers of classes')
parser.add_argument('--gpu', type=int, default=0, help='gpu no.')
parser.add_argument('--dataset', type=str, default='mnist',
                    choices=['mnist', 'cifar10'])
opt = parser.parse_args()
print(opt)

torch.cuda.set_device(opt.gpu)
cuda = True if torch.cuda.is_available() else False 
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
TensorL = torch.cuda.LongTensor if cuda else torch.LongTensor

# ----------
#  Training
# ----------
epoch = 0
i = 0


def calc_gradient_penalty(netD, real_data, fake_data):
    BATCH_SIZE = real_data.shape[0]
    if real_data.dim == 2:
        alpha = torch.rand(BATCH_SIZE, 1)
    else:
        alpha = torch.rand(BATCH_SIZE, 1, 1)

    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(
                                  disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1))**12).mean()

    return gradient_penalty


def calculate_loss(label_index):
    temp_class_index = label_index
    finished_epoch = 0
    loss_pen = 0
    mainloss_p = 0
    imgs = training_iterator[temp_class_index].__next__()
    if imgs is None:
        return 0, -1, 0
    finished_epoch = training_iterator[temp_class_index]._finished_epoch

    positive_data = Tensor(imgs[0])
    loss_pen = loss_pen + calc_gradient_penalty(modelMain[temp_class_index], positive_data, positive_data)
    score_temp_0 = modelMain[temp_class_index].forward(positive_data)
    mainloss_p = mainloss_p + torch.log(torch.sigmoid(1 * score_temp_0) + 1e-2).mean()
    
    loss = - 1.0 * mainloss_p + 0.1 * loss_pen
    optimizer[temp_class_index].zero_grad()
    loss.backward()
    optimizer[temp_class_index].step()

    return loss_pen.data, finished_epoch, mainloss_p.data
# ####################################################################################################

OCDataset = helper.load_dataset(opt.dataset)
oc_dataset = OCDataset(opt)
labels = oc_dataset.labels
num_class = oc_dataset.num_classes
modelMain = {}
optimizer = {}
training_iterator = {}
for label_index in range(num_class):
    if opt.dataset == 'mnist':
        modelMain[label_index] = Recognizer_mlp(opt)
    elif opt.dataset == 'cifar10':
        modelMain[label_index] = Recognizer_mlp_cifar(opt)
    if cuda:
        modelMain[label_index].cuda()
    training_iterator[label_index] = oc_dataset.get_training_iterator(label_index)


opt.class_groups = [1] * 10
opt.task_bounds = np.array([0] + opt.class_groups).cumsum()
for label_index in range(num_class):
    print('-----------------------------', label_index)
    max_auc = 0
    epoch_num = 0

    optimizer[label_index] = torch.optim.SGD(
            modelMain[label_index].parameters(), lr=0.1, momentum=0.9)

    while True:
        i += 1
        loss, finished_epoch, loss1 = calculate_loss(label_index)

        if finished_epoch == -1:
            break
        #############################################################################
        # Begin testing the learned model.
        if finished_epoch != epoch:
            epoch = finished_epoch
            i = 0
            if (epoch_num + 1) % (opt.max_epochs / opt.max_epochs) == 0:
                for label_test in range(label_index + 1):
                    modelMain[label_test].eval()

                testiterator = oc_dataset.get_testing_iterator()
                score_all = []
                Y_all = []
                for test_imgs in testiterator:
                    
                    test_imgs_data = Tensor(test_imgs[0])

                    test_imgs_f = test_imgs_data
                    test_imgs_label = Tensor(test_imgs[1])
                    
                    score_temp = modelMain[label_index].forward(test_imgs_f)
                    score_temp = torch.sigmoid(score_temp)
                    score_all.append(score_temp.squeeze())
                    Y_all.append(test_imgs_label)

                score_all = torch.cat(tuple(score_all), 0)
                Y_all = torch.cat(tuple(Y_all), 0)
                Y_all = 1 - (torch.abs(torch.sign(Y_all - label_index)))
                auc_value = roc_auc_score(Y_all.cpu().detach(), score_all.cpu().detach())

                modelMain[label_index].train()
                max_auc = max(max_auc, auc_value)
                #print("[Epoch %d/%d/%d] [Loss1: %0.2f] [PenP loss: %0.2f] [AUC: %0.4f]" %
                #    (finished_epoch + 1, opt.max_epochs, label_index + 1, loss1, loss, auc_value)) 
                if (epoch_num + 1) % (opt.max_epochs / 1) == 0:
                    print("[Epoch %d/%d/%d] [Loss1: %0.2f] [PenP loss: %0.2f] [AUC: %0.4f]" %
                        (finished_epoch + 1, opt.max_epochs, label_index + 1, loss1, loss, max_auc))

            epoch_num += 1
        i = i+1


