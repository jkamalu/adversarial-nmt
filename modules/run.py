from collections import defaultdict

import torch
from torch.nn import ModuleDict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from modules.model import Encoder, Discriminator, BidirectionalTranslator
from modules.metrics import loss_fn, accuracy, exact_match, sentence_bleu, write_to_tensorboard
from modules.utils import init_output_dirs, save_checkpoint

def switch(step, config):
    """
    Return False if training discriminator, True otherwise
    """
    if len(config["regularization"]["type"]) == 0:
        return True
    if step % 2 == 0 and step % 100 >= config["discriminator_kwargs"]["warmup"]:
        return True
    return False

def switch_trainable(model, step, config):
    switch_ = switch(step, config)
    if len(model.discriminators) > 0:
        for module in model.children():
            if type(module) == ModuleDict:
                for param in module.parameters():
                    param.requires_grad = not switch_
            else:
                for param in module.parameters():
                    param.requires_grad = switch_

def train(model, gen_optimizer, dis_optimizer, dataset_train, dataset_valid, batch_i, config):
    model.train()

    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=config["batch_size"], 
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )

    dataloader_valid = DataLoader(
        dataset_valid, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        pin_memory=True,
        num_workers=4
    )

    writer = SummaryWriter(config["runs_dir"])

    batch_0 = batch_i
    for batch in dataloader_train:

        # Unpack the batch and move to device
        batch_l1, batch_l2 = batch
        sents_l1, sents_no_eos_l1, lengths_l1 = map(lambda t: t.cuda(), batch_l1)
        sents_l2, sents_no_eos_l2, lengths_l2 = map(lambda t: t.cuda(), batch_l2)

        # Clear optimizer
        gen_optimizer.zero_grad()
        dis_optimizer.zero_grad()

        # Alternate trainable for encoder and discriminator parameters
        # switch_trainable(model, batch_i, config)

        # Unpack the batch, run the encoders, run the decoders
        sents_l1, sents_l2, enc_out_l1, enc_out_l2, dec_out_l1, dec_out_l2 = model(
            sents_l1, sents_no_eos_l1, lengths_l1,
            sents_l2, sents_no_eos_l2, lengths_l2
        )

        # Initial default values for regularization 
        real_pred_l1 = {}
        real_pred_l2 = {}
        switch_ = switch(batch_i, config)
        label = lambda x: 0.9 if x else 0.1
        real_l1 = torch.tensor([label(switch_)] * config["batch_size"]).unsqueeze(-1).cuda()
        real_l2 = torch.tensor([label(not switch_)] * config["batch_size"]).unsqueeze(-1).cuda() 

        # Gather hidden discriminator labels/predictions
        if "hidden" in config["regularization"]["type"]:
            # Use the pooled outputs of the encoders for regularization
            pred_hddn_l1 = model.discriminate("hidden", enc_out_l1)
            pred_hddn_l2 = model.discriminate("hidden", enc_out_l2)
            real_pred_l1["hidden"] = real_l1, pred_hddn_l1
            real_pred_l2["hidden"] = real_l2, pred_hddn_l2

        loss, loss_l2, loss_l1, reg_loss_l1, reg_loss_l2 = loss_fn(
            sents_l1[:, 1:], sents_l2[:, 1:],
            dec_out_l1, dec_out_l2,
            real_pred_l1, real_pred_l2,
            ignore_index_l1=dataset_train.tokenizer_l1.pad_token_id,
            ignore_index_l2=dataset_train.tokenizer_l2.pad_token_id
        )

        acc_l1 = accuracy(real_pred_l1)
        acc_l2 = accuracy(real_pred_l2)

        # Optimize trainable parameters
        loss.backward()
        if switch_:
            gen_optimizer.step()
        else:
            dis_optimizer.step()

        batch_i += 1
        if batch_i > batch_0: 

            # Write training losses/metrics to stdout and tensorboard
            if batch_i % config["log_frequency"] == 0:
                
                print("Step {}: loss {:.4}\n\tcce-l1 {:.4}, cce-l2 {:.4}\n\t\tbce-hddn-l1 {:.4}, bce-hddn-l2 {:.4}\n\t\t\tacc-hddn-l1 {:.4}, acc-hddn-l2 {:.4}".format(
                        str(batch_i).zfill(6), 
                        loss.item(), 
                        loss_l1.item(), loss_l2.item(),
                        reg_loss_l1["hidden"].item(), reg_loss_l2["hidden"].item(),
                        acc_l1["hidden"].item(), acc_l2["hidden"].item()
                    )
                )
                cce_metrics = {"l1-l2": loss_l2.item(), "l2-l1": loss_l1.item()}
                write_to_tensorboard("CCE", cce_metrics, training=True, step=batch_i, writer=writer)
                bce_metrics = {"hidden l1": reg_loss_l1["hidden"].item(), "hidden l2": reg_loss_l2["hidden"].item()}
                write_to_tensorboard("BCE", bce_metrics, training=True, step=batch_i, writer=writer)                
                acc_metrics = {"accuracy l1": acc_l1["hidden"].item(), "accuracy l2": acc_l2["hidden"].item()}
                write_to_tensorboard("ACC", acc_metrics, training=True, step=batch_i, writer=writer)

            # Write validation losses/metrics to stdout and tensorboard
            if batch_i % config["val_frequency"] == 0:
                metrics_dict = valid(model, dataloader_valid, config)
                metrics_dict_tb = lambda key: {
                    "l1-l2": metrics_dict["l1-l2"][key],
                    "l2-l1": metrics_dict["l2-l1"][key]
                }
                write_to_tensorboard("CCE", metrics_dict_tb("loss"), training=False, step=batch_i, writer=writer)
                write_to_tensorboard("BLEU", metrics_dict_tb("bleu"), training=False, step=batch_i, writer=writer)
                write_to_tensorboard("EM", metrics_dict_tb("em"), training=False, step=batch_i, writer=writer)

            # Save weights and terminate training
            if batch_i >= config["n_train_steps"]:
                save_checkpoint(model, gen_optimizer, dis_optimizer, batch_i, config["experiment"])
                break

            # Save weights and continue training
            if batch_i % config["checkpoint_frequency"] == 0:
                save_checkpoint(model, gen_optimizer, dis_optimizer, batch_i, config["experiment"])

def evaluate(model, dataset, config):
    model.eval()
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        pin_memory=True,
        num_workers=4
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

    if is_training or not config["do_full_eval"]:
        n_valid_steps = config["n_valid_steps"]
    else:
        n_valid_steps = config["n_valid"]

    with torch.no_grad():

        for batch_j, batch in enumerate(dataloader):

            if batch_j >= n_valid_steps:
                break

            batch_l1, batch_l2 = batch
            sents_l1, sents_no_eos_l1, lengths_l1 = map(lambda t: t.cuda(), batch_l1)
            sents_l2, sents_no_eos_l2, lengths_l2 = map(lambda t: t.cuda(), batch_l2)

            sents_l1, sents_l2, enc_out_l1, enc_out_l2, dec_out_l1, dec_out_l2 = model(
                sents_l1, sents_no_eos_l1, lengths_l1,
                sents_l2, sents_no_eos_l2, lengths_l2
            )

            loss, loss_l2, loss_l1, reg_loss_l1, reg_loss_l2 = loss_fn(
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
            for idx in range(config["batch_size"]):
                text_real_l2 = dataloader.dataset.tokenizer_l2.decode(sents_l2[idx, 1:-1].tolist())
                text_pred_l2 = dataloader.dataset.tokenizer_l2.decode(preds_l2[idx, 0:-1].tolist())
                metrics_dict["l1-l2"]["bleu"][-1].append(sentence_bleu([text_real_l2], text_pred_l2))
                text_real_l1 = dataloader.dataset.tokenizer_l1.decode(sents_l1[idx, 1:-1].tolist())
                text_pred_l1 = dataloader.dataset.tokenizer_l1.decode(preds_l1[idx, 0:-1].tolist())
                metrics_dict["l2-l1"]["bleu"][-1].append(sentence_bleu([text_real_l1], text_pred_l1))
            metrics_dict["l1-l2"]["bleu"][-1] = sum(metrics_dict["l1-l2"]["bleu"][-1]) / \
                                                len(metrics_dict["l1-l2"]["bleu"][-1])
            metrics_dict["l2-l1"]["bleu"][-1] = sum(metrics_dict["l2-l1"]["bleu"][-1]) / \
                                                len(metrics_dict["l2-l1"]["bleu"][-1])

            metrics_dict["l1-l2"]["em"].append(exact_match(sents_l2[:, 1:-1], preds_l2[:, 0:-1], ignore_index=dataloader.dataset.tokenizer_l2.pad_token_id))
            metrics_dict["l2-l1"]["em"].append(exact_match(sents_l1[:, 1:-1], preds_l1[:, 0:-1], ignore_index=dataloader.dataset.tokenizer_l1.pad_token_id))

    print("Ex.")
    print("l1 real: {}".format(text_real_l1))
    print("l1 pred: {}".format(text_pred_l1))
    print("l2 real: {}".format(text_real_l2))
    print("l2 pred: {}".format(text_pred_l2))

    if is_training:
        model.train()

    for language in metrics_dict:
        for metric in metrics_dict[language]:
            metrics_dict[language][metric] = sum(metrics_dict[language][metric]) / n_valid_steps

    print("Loss: l1-l2 {:.3}\tl2-l1 {:.3}".format(metrics_dict["l1-l2"]["loss"], metrics_dict["l2-l1"]["loss"]))
    print("BLEU: l1-l2 {:.3}\tl2-l1 {:.3}".format(metrics_dict["l1-l2"]["bleu"], metrics_dict["l2-l1"]["bleu"]))
    print("EM  : l1-l2 {:.3}\tl2-l1 {:.3}".format(metrics_dict["l1-l2"]["em"], metrics_dict["l2-l1"]["em"]))

    return metrics_dict
