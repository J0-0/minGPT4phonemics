"""
Trains a character-level language model.
"""

import os
import sys
import csv
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from model import GPT
from trainer import Trainer
from utils import set_seed, setup_logging, CfgNode as CN
import torch
torch.cuda.empty_cache()
#print(torch.cuda.memory_summary(device=None, abbreviated=False))
# -----------------------------------------------------------------------------

# BOOL TRAINING TYPES
add_layer = True
model_name = 'gpt2'
model_title = "model_pretrained_gpt2_added_layer"
# -----------------------------------------------------------------------------


def get_config() :
    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    #C.model.model_type = 'gpt-mini'
    C.model.model_type = model_name

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4  # the model we're using is so small that we can go a bit faster

    return C


# -----------------------------------------------------------------------------

class CharDataset(Dataset) :
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config() :
        C = CN()
        C.block_size = 128
        return C

    def __init__(self, config, data) :
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = {ch : i for i, ch in enumerate(chars)}
        self.itos = {i : ch for i, ch in enumerate(chars)}
        self.vocab_size = vocab_size
        self.data = data
        print("DICO", self.stoi)

    def get_vocab_size(self) :
        return self.vocab_size

    def get_block_size(self) :
        return self.config.block_size

    def __len__(self) :
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx) :
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx :idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1 :], dtype=torch.long)
        return x, y


# -----------------------------------------------------------------------------

if __name__ == '__main__' :

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1 :])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    # text = open('/content/drive/MyDrive/wiki_train_to_phonemes.txt', 'r').read() #/content/minGPT4phonemics/mingpt/
    # text = open('/content/drive/MyDrive/stimuli_minGPT/wikitext-103/wiki_test_to_phonemes_punct.txt', 'r').read()
    text = open('/content/drive/MyDrive/stimuli_minGPT/total_train_wiki103_ph_ponct.txt', 'r').read()
    train_dataset = CharDataset(config.data, text)
    path_drive_pretrained = "/content/drive/MyDrive/out/chargpt/"+ model_title +".pt"

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()  # 51
    voc_size = config.model.vocab_size
    config.model.block_size = train_dataset.get_block_size()
    model = GPT.from_pretrained(model_name, add_layer = add_layer)
    print("config.model.vocab_size, config.model.block_size", config.model.vocab_size, config.model.block_size)
    # config.model.vocab_size, config.model.block_size 72 128 #75 128
    #model = GPT(config.model)
    want_pretrained_model = True
    if want_pretrained_model:
        model.load_state_dict(torch.load(path_drive_pretrained))
    print("model =", model)
    # config.model  model_type: gpt-mini

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # iteration callback
    def batch_end_callback(trainer) :

        if trainer.iter_num % 10 == 0 :
            print(
                f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0 :
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad() :
                # sample from the model...
                context = "əʊ ɡɑd əʊ ɡɑd"
                x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None, ...].to(
                    trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10, add_layer = add_layer)[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])
                print(completion)
                print("config.model ", config.model)
            # chars_computes = sorted(list(set(data)))
            # self.itos = { i:ch for i,ch in enumerate(chars) }
            # train_dataset.itos[int(idx_next)],
            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), path_drive_pretrained)
            # revert model to training mode
            model.train()


    train_bool = False
    if train_bool :
        trainer.set_callback('on_batch_end', batch_end_callback)
        # run the optimization
        trainer.run()

    test_bool = True
    if test_bool :
        # path_save_results = "/content/minGPT4phonemics/results_context"
        # ADD for loop for all 4 files
        path_save_results = "/content/drive/MyDrive/minGPT_results/" + model_title + "/gpt2_willow"
        text_results_acc_for_context = open(path_save_results + "_acc_for_context" + ".txt", "w")
        text_results_acc_for_context.write(
            "size_context" + "   " + "sum_correct_pred =" + "   " + "sum_approx_correct_pred" + "\n")
        model.load_state_dict(torch.load(path_drive_pretrained))
        with open(path_save_results + '_acc_for_context.csv', 'w') as acc_for_context :
            acc_for_context.write("%s,%s,%s\n" % ("size_content", "sum_correct_pred",
                                                  "sum_approx_correct_pred"))
            for size_context in [1, 2, 3, 5, 10, 20, 30, 40, 50, 100, 500, 1000]:
                sum_correct_pred = 0
                sum_approx_correct_pred = 0
                text_test = open('/content/minGPT4phonemics/bids_anonym_stimuli_text/the_black_willow_ph_punct.txt',
                                 'r').read()
                path_save_results_context = path_save_results + "_" + str(size_context)
                # print("train_dataset.stoi", train_dataset.stoi)
                dic_phonemes_pred_proba = {}
                print(len(sorted(list(set(text_test)))), "different characters")
                n_char2predict = len(text_test) - size_context
                print(n_char2predict, " iterations")
                if os.path.exists(path_save_results_context + '.txt') :
                    os.remove(path_save_results_context + '.txt')
                if os.path.exists(path_save_results_context + '.csv') :
                    os.remove(path_save_results_context + '.csv')
                text_results = open(path_save_results_context + '.txt', "w")
                with open(path_save_results_context + '.csv', 'w') as f :
                    f.write("%s,%s,%s,%s\n" % ("ORDER", "predicted CHARACTER",
                                               "target CHARACTER", "PROBABILITY"))
                    for pred_char in range(n_char2predict) :
                        context = text_test[pred_char :size_context + pred_char]
                        test_torch = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[
                            None, ...].to(trainer.device)
                        results = model.generate4testing(test_torch, 1, temperature=1.0, do_sample=True, top_k=10,
                                                 return_proba=True, add_layer = add_layer)
                        result_char, results_probas = results[0][0].tolist()[-1], list(results[1].values())[0]
                        char_result = train_dataset.itos[result_char] if result_char < len(train_dataset.itos) else "_"
                        text_results.write(char_result)
                        sum_correct_pred += (char_result == text_test[size_context + pred_char])
                        range_approx = 3
                        for approx in range(pred_char - range_approx, pred_char + range_approx) :
                            if size_context + approx < len(text_test) :
                                if (char_result == text_test[size_context + approx]) :
                                    sum_approx_correct_pred += 1
                        f.write("%s,%s,%s,%s\n" % (pred_char, result_char,
                                                   train_dataset.stoi[text_test[size_context + pred_char]],
                                                   results_probas))
                        # if pred_char != 0 and pred_char%1000 == 0:
                        # print("pred_char =", pred_char)
                        # print("sum_correct_pred =", np.round(sum_correct_pred/pred_char, 4))
                        # print("sum_approx_correct_pred =",
                        # np.round(sum_approx_correct_pred/pred_char, 4))
                sum_correct_pred = np.round(sum_correct_pred / n_char2predict, 4)
                sum_approx_correct_pred = np.round(sum_approx_correct_pred / n_char2predict, 4)
                print(size_context)
                print("sum_correct_pred =", sum_correct_pred)
                print("sum_approx_correct_pred =", sum_approx_correct_pred)
                text_results_acc_for_context.write(str(size_context) + "   "
                                                   + str(sum_correct_pred) + "   " + str(
                    sum_approx_correct_pred) + "\n")
                acc_for_context.write("%s,%s,%s\n" % (size_context, sum_correct_pred,
                                                      sum_approx_correct_pred))
                text_results.close()
                # text_results_acc_for_context.flush()
        text_results_acc_for_context.close()
