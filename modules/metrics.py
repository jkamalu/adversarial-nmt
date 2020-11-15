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
    
    agg = 0
    
    loss_l1 = cce_l1(pred_l1.transpose(1,2), real_l1)
    agg = agg + loss_l1
    
    loss_l2 = cce_l2(pred_l2.transpose(1,2), real_l2)
    agg = agg + loss_l2
    
    bce_loss_l1 = defaultdict(lambda: torch.tensor(0.0))
    for regularization in real_pred_l1:
        real_l1, pred_l1 = real_pred_l1[regularization]
        bce_loss_l1[regularization] = bce(pred_l1, real_l1)
        agg = agg + bce_loss_l1[regularization]

    bce_loss_l2 = defaultdict(lambda: torch.tensor(0.0))
    for regularization in real_pred_l2:
        real_l2, pred_l2 = real_pred_l2[regularization]
        bce_loss_l2[regularization] = bce(pred_l2, real_l2)
        agg = agg + bce_loss_l2[regularization]

    return agg, loss_l2, loss_l1, bce_loss_l1, bce_loss_l2


def accuracy(real_pred_ys):
    with torch.no_grad():
        reg_accuracies = defaultdict(lambda: torch.tensor(0.0))
        for regularization in real_pred_ys:
            real_y, pred_y = real_pred_ys[regularization]
            pred_y = (torch.sigmoid(pred_y) > 0.5)
            real_y = (real_y > 0.5)
            try:
                reg_accuracies[regularization] = torch.mean(torch.eq(pred_y, real_y).float())
            except:
                continue
        return reg_accuracies

def soft_loss_fn(real_l1, real_l2, pred_l1, pred_l2, en_fr_enc_w, fr_en_enc_w, ignore_index_l1=1, ignore_index_l2=1):
    '''
    Adversarial Loss: standard loss with binary cross entropy on top of the discriminator outputs
    '''
    cce_l1 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index_l1)
    cce_l2 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index_l2)
    bce = torch.nn.BCEWithLogitsLoss()
    
    agg = 0
    
    loss_l1 = cce_l1(pred_l1.transpose(1,2), real_l1)
    agg = agg + loss_l1
    
    loss_l2 = cce_l2(pred_l2.transpose(1,2), real_l2)
    agg = agg + loss_l2
    

    soft_param_share_loss = 0
    count = 0
    for each in en_fr_enc_w:
        if en_fr_enc_w[each].dtype == torch.float32 and en_fr_enc_w[each].shape == fr_en_enc_w[each].shape:
            if 'embedding' not in each:
                soft_param_share_loss += torch.dist(en_fr_enc_w[each], fr_en_enc_w[each],2)

    soft_param_share_loss /= pow(10,3)
    soft_param_share_loss /= 2
    agg = agg + soft_param_share_loss

    return agg, loss_l2, loss_l1, soft_param_share_loss

def exact_match(real, pred, ignore_index=1):
    '''
    Evaluate percent exact match between predictions and ground truth
    '''
    mask = real != ignore_index
    return torch.sum((pred == real) * mask).item() / torch.sum(mask).item()