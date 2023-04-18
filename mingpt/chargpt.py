"""
Trains a character-level language model.
"""

import os
import sys

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
    text = open('/content/minGPT4phonemics/mingpt/wiki_valid_to_phonemes.txt', 'r').read()
    train_dataset = CharDataset(config.data, text)
    # print("DICTIONNARY ", train_dataset.itos)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size() #51
    config.model.block_size = train_dataset.get_block_size()
    print("config.model.vocab_size, config.model.block_size", config.model.vocab_size, config.model.block_size)
    # config.model.vocab_size, config.model.block_size 72 128 #75 128
    model = GPT(config.model)
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


    train_bool = True
    if train_bool:
        trainer.set_callback('on_batch_end', batch_end_callback)

        # run the optimization
        trainer.run()

    test_bool = True
    if test_bool:
        #PATH = "/content/out/chargpt/model_loss0_5.pt" 
        PATH = "/content/drive/MyDrive/out/chargpt/model.pt" #model_loss_0_55.pt
        model.load_state_dict(torch.load(PATH))
        text_test = open('/content/minGPT4phonemics/mingpt/wiki_test_to_phonemes.txt', 'r').read()
        dic_phonemes_pred_proba = {}
        #for i in len(text_test):
        #print(text_test, len(text_test), text_test[:10])
        print("LEN TEXT TEST", len(text_test)) #1285865
        # print(len(sorted(list(set(text_test[:5000]))))) #54 #45 if text_test[:5000]
        print(len(sorted(list(set(text_test))))) #54 #45 if text_test[:5000]
        #print([train_dataset.stoi[s] for s in text_test[:5000]], len([train_dataset.stoi[s] for s in text_test[:5000]])) 
        print([train_dataset.stoi[s] for s in text_test], len([train_dataset.stoi[s] for s in text_test[:5000]])) 
        test_torch = torch.tensor([train_dataset.stoi[s] for s in text_test], dtype=torch.long)[None,...].to(trainer.device)
        print("test_torch ", test_torch.size())
        results = model.generate(test_torch, 500, temperature=1.0, do_sample=True, top_k=10)[0]
        text_results = open('/content/results.txt', "w")
        print("RESULTS ", results.tolist())
        print("RESULTS TRANSLATED ", [train_dataset.itos[i] for i in results.tolist()])
        text_results.write(results.tolist())
        text_results.write([train_dataset.itos[i] for i in results.tolist()])
        test_results.close()

        ## REMOVE text_test[:5000]
