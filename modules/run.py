from collections import defaultdict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from modules.data import Collator
from modules.model import Encoder, Discriminator, BidirectionalTranslator
from modules.metrics import soft_loss_fn, loss_fn, exact_match, sentence_bleu, write_to_tensorboard
from modules.utils import init_output_dirs

import os


def switch_trainable(model, step):    
    switch = step % 2 == 0
    if len(model.discriminators) > 0:
        for module in model.children():
            if type(module) == Discriminator:
                for param in module.parameters():
                    param.requires_grad = not switch
            elif type(module) == Encoder:
                for param in module.parameters():
                    param.requires_grad = switch


def save_checkpoint(model, optimizer, step, ckpt_dir):
    ckpt = "{0}-{1:e}.pt".format(str(step).zfill(6), step)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
    }, os.path.join(ckpt_dir, ckpt))
    

def load_checkpoint(model, optimizer, step, loss, ckpt_dir):
    ckpt = "{0}-{1:e}.pt".format(str(step).zfill(6), step)
    
    state_dict = torch.load(os.path.join(ckpt_dir, ckpt))
    
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return model, optimizer

    
def train(model, dataset_train, dataset_valid, config):    
    collate_fn = Collator(config["maxlen"])
    
    model.cuda()
    
    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=config["batch_size"], 
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn
    )

    dataloader_valid = DataLoader(
        dataset_valid, 
        config["batch_size"], 
        shuffle=True, 
        pin_memory=True,
        collate_fn=collate_fn
    )

    writer = SummaryWriter(config["runs_dir"])
    
    optimizer = Adam(model.parameters(), **config["adam"])
    
    for batch_i, batch in enumerate(dataloader_train):

        # Unpack batch and move to device
        batch_en, batch_fr = batch
        sents_en, sents_no_eos_en, lengths_en = map(lambda t: t.cuda(), batch_en)
        sents_fr, sents_no_eos_fr, lengths_fr = map(lambda t: t.cuda(), batch_fr)

        # Clear optimizer
        optimizer.zero_grad()

        # Alternate trainable for encoder and discriminator parameters
        switch_trainable(model, batch_i)

        # Save weights and continue training
        if batch_i > 0 and batch_i % config["checkpoint_frequency"] == 0:
            save_checkpoint(model, optimizer, batch_i, './saved_model_no_soft')

        # Save weights and terminate training
        if batch_i > 0 and batch_i >= config["max_step_num"]:
            save_checkpoint(model, optimizer, batch_i,'./saved_model_no_soft')
            break

        # Unpack the batch, run the encoders, run the decoders
        sents_en, sents_fr, enc_out_en, enc_out_fr, dec_out_en, dec_out_fr, en_fr_enc_w, fr_en_enc_w = model(
            sents_en, sents_no_eos_en, lengths_en,
            sents_fr, sents_no_eos_fr, lengths_fr
        )

        # Initial default values for regularization 
        real_pred_ys = {}
        switch = batch_i % 2 == 0
        y_real = [float(switch)] * config["batch_size"] + [float(not switch)] * config["batch_size"]
        y_real = torch.tensor(y_real).unsqueeze(-1).cuda()

        # Gather hidden discriminator labels/predictions
        if "hidden" in config["regularization"]["type"]:
            # Use the pooled outputs of the encoders for regularization
            y_hddn_pred_en = model.discriminate("hidden", enc_out_en)
            y_hddn_pred_fr = model.discriminate("hidden", enc_out_fr)
            y_hddn_pred = torch.cat([y_hddn_pred_en, y_hddn_pred_fr])
            real_pred_ys["hidden"] = y_real, y_hddn_pred
        
        '''
        loss, loss_en2fr, loss_fr2en, reg_losses = loss_fn(
            sents_en[:, 1:], sents_fr[:, 1:],
            dec_out_en, dec_out_fr, 
            real_pred_ys,
            ignore_index=1
        )
        '''

        loss, loss_en2fr, loss_fr2en, soft_param_share_loss = soft_loss_fn(
            sents_en[:, 1:], sents_fr[:, 1:],
            dec_out_en, dec_out_fr, 
            en_fr_enc_w,
            fr_en_enc_w,
            ignore_index=1
        )

        # Optimize trainable parameters
        loss.backward()
        optimizer.step()

        # Write training losses/metrics to stdout and tensorboard
        if batch_i > 0 and batch_i % config["log_frequency"] == 0:
            '''
            print("Step {}: loss {}, en-fr {}, fr-en {}, hddn {}".format(
                str(batch_i).zfill(6), loss.item(), loss_en2fr.item(), loss_fr2en.item(), reg_losses["hidden"].item()))
            '''
            print("Step {}: loss {}, en-fr {}, fr-en {}, hddn {}".format(
                str(batch_i).zfill(6), loss.item(), loss_en2fr.item(), loss_fr2en.item(), soft_param_share_loss.item()))

            cce_metrics = {"en-fr": loss_en2fr.item(), "fr-en": loss_fr2en.item()}
            write_to_tensorboard("CCE", cce_metrics, training=True, step=batch_i, writer=writer)

            bce_metrics = {"soft": soft_param_share_loss.item()}
            write_to_tensorboard("BCE", bce_metrics, training=True, step=batch_i, writer=writer)
        
        # Write validation losses/metrics to stdout and tensorboard
        if batch_i > 0 and batch_i % config["val_frequency"] == 0:
            metrics_dict = valid(model, dataloader_valid, config)
            
            metrics_dict_tb = lambda key: {
                "en-fr": metrics_dict["en-fr"][key],
                "fr-en": metrics_dict["fr-en"][key]
            }

            write_to_tensorboard("CCE", metrics_dict_tb("loss"), training=False, step=batch_i, writer=writer)
            write_to_tensorboard("BLEU", metrics_dict_tb("bleu"), training=False, step=batch_i, writer=writer)
            write_to_tensorboard("EM", metrics_dict_tb("em"), training=False, step=batch_i, writer=writer)


def evaluate(model, dataset, config):
    
    model.cuda()
    
    dataloader = DataLoader(
        dataset, 
        config["batch_size"], 
        shuffle=True, 
        pin_memory=True,
        collate_fn=Collator(config["maxlen"])
    )

    valid(model, dataloader, config)


def valid(model, dataloader, config):
    is_training = model.training

    if is_training:
        model.eval()
        
    metrics_dict = {
        "fr-en": defaultdict(list),
        "en-fr": defaultdict(list)
    }

    with torch.no_grad():

        for batch_j, batch in enumerate(dataloader):
            
            if (batch_j == config["n_valid"]):
                break
            
            batch_en, batch_fr = batch
            sents_en, sents_no_eos_en, lengths_en = map(lambda t: t.cuda(), batch_en)
            sents_fr, sents_no_eos_fr, lengths_fr = map(lambda t: t.cuda(), batch_fr)

            sents_en, sents_fr, enc_out_en, enc_out_fr, dec_out_en, dec_out_fr, en_fr_enc_w, fr_en_enc_w = model(
                sents_en, sents_no_eos_en, lengths_en,
                sents_fr, sents_no_eos_fr, lengths_fr
            )
            
            '''
            loss_all, loss_fr, loss_en, loss_reg = loss_fn(
                sents_en[:, 1:], sents_fr[:, 1:],
                dec_out_en, dec_out_fr,
                ignore_index=1
            )
            '''

            loss_all, loss_fr, loss_en, soft_param_share_loss = soft_loss_fn(
                sents_en[:, 1:], sents_fr[:, 1:],
                dec_out_en, dec_out_fr, 
                en_fr_enc_w,
                fr_en_enc_w,
                ignore_index=1
            )

            metrics_dict["en-fr"]["loss"].append(loss_fr.item())
            metrics_dict["fr-en"]["loss"].append(loss_en.item())

            preds_fr = torch.argmax(dec_out_fr, dim=-1)
            preds_en = torch.argmax(dec_out_en, dim=-1)

            metrics_dict["en-fr"]["bleu"].append([])
            metrics_dict["fr-en"]["bleu"].append([])
            for idx in range(config["batch_size"]):
                text_real_fr = dataloader.dataset.tokenizer_fr.decode(sents_fr[idx, 1:-1].tolist())
                text_pred_fr = dataloader.dataset.tokenizer_fr.decode(preds_fr[idx, 0:-1].tolist())
                metrics_dict["en-fr"]["bleu"][-1].append(sentence_bleu([text_real_fr], text_pred_fr))
                text_real_en = dataloader.dataset.tokenizer_en.decode(sents_en[idx, 1:-1].tolist())
                text_pred_en = dataloader.dataset.tokenizer_en.decode(preds_en[idx, 0:-1].tolist())
                metrics_dict["fr-en"]["bleu"][-1].append(sentence_bleu([text_real_en], text_pred_en))
            metrics_dict["en-fr"]["bleu"][-1] = sum(metrics_dict["en-fr"]["bleu"][-1]) \
                                                    / len(metrics_dict["en-fr"]["bleu"][-1])
            metrics_dict["fr-en"]["bleu"][-1] = sum(metrics_dict["fr-en"]["bleu"][-1]) \
                                                    / len(metrics_dict["fr-en"]["bleu"][-1])

            metrics_dict["en-fr"]["em"].append(exact_match(preds_fr[:, 0:-1], sents_fr[:, 1:-1]))
            metrics_dict["fr-en"]["em"].append(exact_match(preds_en[:, 0:-1], sents_en[:, 1:-1]))

    if is_training:
        model.train()
        
    for language in metrics_dict:
        for metric in metrics_dict[language]:
            metrics_dict[language][metric] = sum(metrics_dict[language][metric]) / config["n_valid"]

    print("Loss: en-fr {:.3},\t fr-en {:.3}".format(metrics_dict["en-fr"]["loss"], metrics_dict["fr-en"]["loss"]))
    print("BLEU: en-fr {:.3},\t fr-en {:.3}".format(metrics_dict["en-fr"]["bleu"], metrics_dict["fr-en"]["bleu"]))
    print("EM  : en-fr {:.3},\t fr-en {:.3}".format(metrics_dict["en-fr"]["em"], metrics_dict["fr-en"]["em"]))
    
    return metrics_dict
