from collections import defaultdict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from modules.model import Encoder, Discriminator, BidirectionalTranslator
from modules.metrics import loss_fn, exact_match, sentence_bleu, write_to_tensorboard
from modules.utils import init_output_dirs


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


def save_checkpoint(model, optimizer, step, loss, ckpt_dir):
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
    
    model.cuda()
    
    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=config["batch_size"], 
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )

    dataloader_valid = DataLoader(
        dataset_valid, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        pin_memory=True
    )

    writer = SummaryWriter(config["runs_dir"])
    
    optimizer = Adam(model.parameters(), **config["adam"])
    
    for batch_i, batch in enumerate(dataloader_train):

        # Unpack the batch and move to device
        batch_l1, batch_l2 = batch
        sents_l1, sents_no_eos_l1, lengths_l1 = map(lambda t: t.cuda(), batch_l1)
        sents_l2, sents_no_eos_l2, lengths_l2 = map(lambda t: t.cuda(), batch_l2)

        # Clear optimizer
        optimizer.zero_grad()

        # Alternate trainable for encoder and discriminator parameters
        switch_trainable(model, batch_i)

        # Save weights and continue training
        if batch_i > 0 and batch_i % config["checkpoint_frequency"] == 0:
            save_weights(model, optimizer, batch_i)

        # Save weights and terminate training
        if batch_i > 0 and batch_i >= config["n_train_steps"]:
            save_weights(model, optimizer, batch_i)
            break

        # Unpack the batch, run the encoders, run the decoders
        sents_l1, sents_l2, enc_out_l1, enc_out_l2, dec_out_l1, dec_out_l2 = model(
            sents_l1, sents_no_eos_l1, lengths_l1,
            sents_l2, sents_no_eos_l2, lengths_l2
        )

        # Initial default values for regularization 
        real_pred_ys = {}
        switch = batch_i % 2 == 0
        y_real = [float(switch)] * config["batch_size"] + [float(not switch)] * config["batch_size"]
        y_real = torch.tensor(y_real).unsqueeze(-1).cuda()

        # Gather hidden discriminator labels/predictions
        if "hidden" in config["regularization"]["type"]:
            # Use the pooled outputs of the encoders for regularization
            y_hddn_pred_l1 = model.discriminate("hidden", enc_out_l1)
            y_hddn_pred_l2 = model.discriminate("hidden", enc_out_l2)
            y_hddn_pred = torch.cat([y_hddn_pred_l1, y_hddn_pred_l2])
            real_pred_ys["hidden"] = y_real, y_hddn_pred

        loss, loss_l2, loss_l1, reg_losses = loss_fn(
            sents_l1[:, 1:], sents_l2[:, 1:],
            dec_out_l1, dec_out_l2, 
            real_pred_ys,
            ignore_index_l1=dataset_train.tokenizer_l1.pad_token_id,
            ignore_index_l2=dataset_train.tokenizer_l2.pad_token_id
        )
                
        # Optimize trainable parameters
        loss.backward()
        optimizer.step()

        # Write training losses/metrics to stdout and tensorboard
        if batch_i > 0 and batch_i % config["log_frequency"] == 0:
            
            print("Step {}: loss {}, l1-l2 {}, l2-l1 {}, hddn {}".format(
                str(batch_i).zfill(6), loss.item(), loss_l2.item(), loss_l1.item(), reg_losses["hidden"].item()))

            cce_metrics = {"l1-l2": loss_l2.item(), "l2-l1": loss_l1.item()}
            write_to_tensorboard("CCE", cce_metrics, training=True, step=batch_i, writer=writer)

            bce_metrics = {"hddn": reg_losses["hidden"].item()}
            write_to_tensorboard("BCE", bce_metrics, training=True, step=batch_i, writer=writer)
        
        # Write validation losses/metrics to stdout and tensorboard
        if batch_i > 0 and batch_i % config["val_frequency"] == 0:
            metrics_dict = valid(model, dataloader_valid, config)
            
            metrics_dict_tb = lambda key: {
                "l1-l2": metrics_dict["l1-l2"][key],
                "l2-l1": metrics_dict["l2-l1"][key]
            }

            write_to_tensorboard("CCE", metrics_dict_tb("loss"), training=False, step=batch_i, writer=writer)
            write_to_tensorboard("BLEU", metrics_dict_tb("bleu"), training=False, step=batch_i, writer=writer)
            write_to_tensorboard("EM", metrics_dict_tb("em"), training=False, step=batch_i, writer=writer)


def evaluate(model, dataset, config):
    
    model.cuda()
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        pin_memory=True
    )

    valid(model, dataloader, config)


def valid(model, dataloader, config):
    is_training = model.training

    if is_training:
        model.eval()
        
    metrics_dict = {
        "l2-l1": defaultdict(list),
        "l1-l2": defaultdict(list)
    }

    with torch.no_grad():

        for batch_j, batch in enumerate(dataloader):
            
            if is_training and batch_j >= config["n_valid_steps"]:
                break
            elif not config["do_full_eval"] and batch_j >= config["n_valid_steps"]:
                break
            else:
                pass
            
            batch_l1, batch_l2 = batch
            sents_l1, sents_no_eos_l1, lengths_l1 = map(lambda t: t.cuda(), batch_l1)
            sents_l2, sents_no_eos_l2, lengths_l2 = map(lambda t: t.cuda(), batch_l2)

            sents_l1, sents_l2, enc_out_l1, enc_out_l2, dec_out_l1, dec_out_l2 = model(
                sents_l1, sents_no_eos_l1, lengths_l1,
                sents_l2, sents_no_eos_l2, lengths_l2
            )
            
            loss, loss_l2, loss_l1, reg_losses = loss_fn(
                sents_l1[:, 1:], sents_l2[:, 1:],
                dec_out_l1, dec_out_l2, 
                ignore_index_l1=dataloader.dataset.tokenizer_l1.pad_token_id,
                ignore_index_l2=dataloader.dataset.tokenizer_l2.pad_token_id
            )

            metrics_dict["l1-l2"]["loss"].append(loss_l2.item())
            metrics_dict["l2-l1"]["loss"].append(loss_l1.item())

            preds_l1 = torch.argmax(dec_out_l1, dim=-1)
            preds_l2 = torch.argmax(dec_out_l2, dim=-1)

            metrics_dict["l1-l2"]["bleu"].append([])
            metrics_dict["l2-l1"]["bleu"].append([])
            for idx in range(config["batch_size"] * config["world_size"]):
                text_real_l2 = dataloader.dataset.tokenizer_l2.decode(sents_l2[idx, 1:-1].tolist())
                text_pred_l2 = dataloader.dataset.tokenizer_l2.decode(preds_l2[idx, 0:-1].tolist())
                metrics_dict["l1-l2"]["bleu"][-1].append(sentence_bleu([text_real_l2], text_pred_l2))
                text_real_l1 = dataloader.dataset.tokenizer_l1.decode(sents_l1[idx, 1:-1].tolist())
                text_pred_l1 = dataloader.dataset.tokenizer_l1.decode(preds_l1[idx, 0:-1].tolist())
                metrics_dict["l2-l1"]["bleu"][-1].append(sentence_bleu([text_real_l1], text_pred_l1))
            metrics_dict["l1-l2"]["bleu"][-1] = sum(metrics_dict["l1-l2"]["bleu"][-1]) \
                                                    / len(metrics_dict["l1-l2"]["bleu"][-1])
            metrics_dict["l2-l1"]["bleu"][-1] = sum(metrics_dict["l2-l1"]["bleu"][-1]) \
                                                    / len(metrics_dict["l2-l1"]["bleu"][-1])

            metrics_dict["l1-l2"]["em"].append(exact_match(preds_l2[:, 0:-1], sents_l2[:, 1:-1], ignore_index=dataloader.dataset.tokenizer_l2.pad_token_id))
            metrics_dict["l2-l1"]["em"].append(exact_match(preds_l1[:, 0:-1], sents_l1[:, 1:-1], ignore_index=dataloader.dataset.tokenizer_l1.pad_token_id))

    if is_training:
        model.train()
        
    for language in metrics_dict:
        for metric in metrics_dict[language]:
            metrics_dict[language][metric] = sum(metrics_dict[language][metric]) / config["n_valid"]

    print("Loss: l1-l2 {:.3},\t l2-l1 {:.3}".format(metrics_dict["l1-l2"]["loss"], metrics_dict["l2-l1"]["loss"]))
    print("BLEU: l1-l2 {:.3},\t l2-l1 {:.3}".format(metrics_dict["l1-l2"]["bleu"], metrics_dict["l2-l1"]["bleu"]))
    print("EM  : l1-l2 {:.3},\t l2-l1 {:.3}".format(metrics_dict["l1-l2"]["em"], metrics_dict["l2-l1"]["em"]))
    
    return metrics_dict
