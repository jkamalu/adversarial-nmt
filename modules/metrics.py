from collections import defaultdict

from nltk.translate.bleu_score import sentence_bleu

import torch


def write_to_tensorboard(base, metrics, training, step, writer):
    """
    Write data to tensorboard.

    Example usage:
        writer = SummaryWriter("runs/regularize_hidden")
        write_to_tensorboard("CCE", {'l1-l2': 0.5, 'l2-l1': 0.4}, True, 42, writer)
    """

    tag = "{}/{}".format(base, "train" if training else "val")

    writer.add_scalars(tag, metrics, step)


def loss_fn(real_l1, real_l2, pred_l1, pred_l2, real_pred_l1={}, real_pred_l2={}, ignore_index_l1=1, ignore_index_l2=1):
    '''
    Adversarial Loss: standard loss with binary cross entropy on top of the discriminator outputs
    '''
    cce_l1 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index_l1)
    cce_l2 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index_l2)
    bce = torch.nn.BCEWithLogitsLoss()
    
    loss_l1 = cce_l1(pred_l1.transpose(1,2), real_l1)
    
    loss_l2 = cce_l2(pred_l2.transpose(1,2), real_l2)
    
    bce_loss_l1 = defaultdict(lambda: torch.tensor(0.0))
    for regularization in real_pred_l1:
        real_l1, pred_l1 = real_pred_l1[regularization]
        bce_loss_l1[regularization] = bce(pred_l1, real_l1)

    bce_loss_l2 = defaultdict(lambda: torch.tensor(0.0))
    for regularization in real_pred_l2:
        real_l2, pred_l2 = real_pred_l2[regularization]
        bce_loss_l2[regularization] = bce(pred_l2, real_l2)
       
    agg = loss_l2 + loss_l1 + \
          torch.sum(torch.tensor(list(bce_loss_l1.values()))) + \
          torch.sum(torch.tensor(list(bce_loss_l2.values())))

    return agg, loss_l2, loss_l1, bce_loss_l1, bce_loss_l2


def accuracy(real_pred_ys):
    reg_accuracies = defaultdict(lambda: torch.tensor(0.0))
    for regularization in real_pred_ys:
        real_y, pred_y = real_pred_ys[regularization]
        print(real_y[0])
        print(pred_y[0])
    return reg_accuracies


def exact_match(real, pred, ignore_index=1):
    '''
    Evaluate percent exact match between predictions and ground truth
    '''
    mask = real != ignore_index
    return torch.sum((pred == real) * mask).item() / torch.sum(mask).item()