from collections import defaultdict

from nltk.translate.bleu_score import sentence_bleu

import torch


def write_to_tensorboard(base, metrics, training, step, writer):
    """
    Write data to tensorboard.

    Example usage:
        writer = SummaryWriter("runs/regularize_hidden")
        write_to_tensorboard("CCE", {'fr-en': 0.5, 'fr-en': 0.4}, True, 42, writer)
    """

    tag = "{}/{}".format(base, "train" if training else "val")

    writer.add_scalars(tag, metrics, step)


def loss_fn(real_en, real_fr, pred_en, pred_fr, real_pred_ys={}, ignore_index=1):
    '''
    Adversarial Loss: standard loss with binary cross entropy on top of the discriminator outputs
    '''
    cce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    loss_en2fr = cce_loss(pred_fr.transpose(1,2), real_fr)
    loss_fr2en = cce_loss(pred_en.transpose(1,2), real_en)
    
    bce_loss = torch.nn.BCEWithLogitsLoss()
    reg_losses = defaultdict(lambda: torch.tensor(0.0))
    for regularization in real_pred_ys:
        real_y, pred_y = real_pred_ys[regularization]
        reg_losses[regularization] = bce_loss(pred_y, real_y)

    return loss_en2fr + loss_fr2en + torch.sum(torch.tensor(list(reg_losses.values()))), loss_en2fr, loss_fr2en, reg_losses

def soft_loss_fn(real_en, real_fr, pred_en, pred_fr, en_fr_enc_w, fr_en_enc_w, ignore_index=1):
    '''
    Adversarial Loss: standard loss with binary cross entropy on top of the discriminator outputs
    '''
    cce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    loss_en2fr = cce_loss(pred_fr.transpose(1,2), real_fr)
    loss_fr2en = cce_loss(pred_en.transpose(1,2), real_en)

    soft_param_share_loss = 0
    count = 0
    for each in en_fr_enc_w:
        if en_fr_enc_w[each].dtype == torch.float32 and en_fr_enc_w[each].shape == fr_en_enc_w[each].shape:
            if 'embedding' not in each:
                soft_param_share_loss += torch.dist(en_fr_enc_w[each], fr_en_enc_w[each],2)

    soft_param_share_loss /= pow(10,3)
    soft_param_share_loss /= 2

    return loss_en2fr + loss_fr2en, loss_en2fr, loss_fr2en, soft_param_share_loss

def exact_match(pred, real, ignore_index=1):
    '''
    Evaluate percent exact match between predictions and ground truth
    '''
    mask = real != ignore_index
    return torch.sum((pred == real) * mask).item() / torch.sum(mask).item()