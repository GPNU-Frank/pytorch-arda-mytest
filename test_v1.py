


import torch.nn as nn
import os

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from models.classifier import  Classifier

from models.discriminator import  Discriminator
from models.generator import  Generator
from models.generator_lower_size import Generator_Lower
from misc import params
from get_data import  *

use_cuda = torch.cuda.is_available()


def test(classifier, generator, generator_lower,data_loader, dataset="MNIST"):
    """Evaluate classifier on source or target domains."""
    # set eval state for Dropout and BN layers
    print(dataset)
    if dataset == "USPS":
        print("change generator")
        generator = generator_lower
    generator.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        if dataset == "USPS":
            images = make_lower_size(images)
        images = Variable(images, volatile=True)

        labels = Variable(labels.squeeze_())
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()

        preds = classifier(generator(images))
        loss += criterion(preds, labels).data[0]

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {:.5f}, Avg Accuracy = {:2.5%}".format(loss, acc))


def init_weights(layer):
    """Init weights for layers."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)







def read_model(net , restore):
    """Init models with cuda and weights."""
    # init weights of model
    net.apply(init_weights)

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net

# load dataset
src_data_loader = get_data_loader(params.src_dataset)
src_data_loader_test = get_data_loader(params.src_dataset, train=False)
tgt_data_loader = get_data_loader(params.tgt_dataset)
tgt_data_loader_test = get_data_loader(params.tgt_dataset, train=False)


#load model
classifier = read_model(Classifier(),restore= params.c_model_restore_v1)
generator = read_model(Generator(),restore=params.g_model_restore_v1)
generator_lower = read_model(Generator_Lower(),restore=params.g_l_model_restore_v1)
# evaluate models
print("=== Evaluating models ===")
print(">>> on source domain <<<")
test(classifier, generator, generator_lower,src_data_loader, params.src_dataset)
print(">>> on target domain <<<")
test(classifier, generator, generator_lower,tgt_data_loader, params.tgt_dataset)