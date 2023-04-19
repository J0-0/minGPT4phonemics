"""
Trains a character-level language model.
"""

import os
import sys
import csv


import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from model import GPT
from trainer import Trainer
from utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C

# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = data
        print("DICO", self.stoi)

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    #text = open('/content/drive/MyDrive/wiki_train_to_phonemes.txt', 'r').read() #/content/minGPT4phonemics/mingpt/
    text = open('/content/drive/MyDrive/stimuli_minGPT/wikitext-103/wiki_test_to_phonemes_punct.txt', 'r').read()
    train_dataset = CharDataset(config.data, text)
    # print("DICTIONNARY ", train_dataset.itos)

    # construct the model
    config.model.vocab_size = 75 #train_dataset.get_vocab_size() #51
    config.model.block_size = train_dataset.get_block_size()
    print("config.model.vocab_size, config.model.block_size", config.model.vocab_size, config.model.block_size)
    # config.model.vocab_size, config.model.block_size 72 128 #75 128
    model = GPT(config.model)
    want_pretrained_model = False
    if want_pretrained_model:
        PATH = "/content/drive/MyDrive/out/chargpt/model.pt" #model_loss_0_55.pt
        model.load_state_dict(torch.load(PATH))
    print("config.model ", config.model)
    #config.model  model_type: gpt-mini

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                context = "əʊ ɡɑd əʊ ɡɑd"
                x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])
                print(completion)
            # chars_computes = sorted(list(set(data)))
            # self.itos = { i:ch for i,ch in enumerate(chars) } 
            # train_dataset.itos[int(idx_next)],
            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()


    train_bool = False
    if train_bool:
        trainer.set_callback('on_batch_end', batch_end_callback)

        # run the optimization
        trainer.run()

    test_bool = True
    if test_bool:
        size_context = 20
        text_test = open('/content/minGPT4phonemics/bids_anonym_stimuli_text/the_black_willow_ph_punct.txt', 'r').read()
        path_save_results = "/content/minGPT4phonemics/results_context_20"
        PATH = "/content/drive/MyDrive/out/chargpt/model.pt" #model_loss_0_55.pt
        model.load_state_dict(torch.load(PATH))
        print("train_dataset.stoi", train_dataset.stoi)
        dic_phonemes_pred_proba = {}
        print(len(sorted(list(set(text_test)))), "different characters")
        n_char2predict = len(text_test)-20
        print(n_char2predict, " iterations")
        text_results = open(path_save_results+'.txt', "w")
        with open(path_save_results+'.csv', 'w') as f:
          f.write("%s,%s,%s,%s\n"%("ORDER", "predicted CHARACTER", 
          "target CHARACTER", "PROBABILITY"))
          for pred_char in range(n_char2predict):
            context = text_test[pred_char:size_context+pred_char]
            test_torch = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
            results = model.generate(test_torch, 1, temperature=1.0, do_sample=True, top_k=10, return_proba = True)
            result_char, results_probas = results[0][0].tolist()[-1], results[1]
            char_results = train_dataset.itos[result_char] if result_char <len(train_dataset.itos) else "_" 
            print(pred_char, char_results, 
            text_test[size_context+pred_char])
            text_results.write(char_results)
            for enum, key in enumerate(results_probas.keys()):
              f.write("%s,%s,%s,%s\n"%(pred_char,key[1], 
              train_dataset.stoi[text_test[size_context+enum]], results_probas[key]))
        text_results.close()
        
