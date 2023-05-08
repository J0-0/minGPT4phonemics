"""
Trains a character-level language model.
"""

import os
import sys
import csv
import numpy as np
import pandas as pd
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
add_layer = False
on_GPU = True
model_name = "gpt-mini" #'gpt2'
model_title = "GPT_mini_untrained" # "model_pretrained_gpt2_added_layer"
model_weights_stored = "GPT_mini_untrained" #"GPT_mini_pretrained"
model_after_training = "GPT_mini_untrained" #"GPT_mini_more_pretrained"

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
    path_drive_pretrained = "/content/drive/MyDrive/out/chargpt/"+ model_weights_stored +".pt"
    path_model_more_trained = "/content/drive/MyDrive/out/chargpt/"+ model_after_training +".pt"
    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()  # 51
    voc_size = config.model.vocab_size
    config.model.block_size = train_dataset.get_block_size()
    if model_name == "gpt-mini":
        model = GPT(config.model)
    else:
        model = GPT.from_pretrained(model_name, add_layer = add_layer)
    print("config.model.vocab_size, config.model.block_size", config.model.vocab_size, config.model.block_size)
    # config.model.vocab_size, config.model.block_size 72 128 #75 128
    want_pretrained_model = False
    if want_pretrained_model:
        if on_GPU:
            model.load_state_dict(torch.load(path_drive_pretrained))
        else:
            model.load_state_dict(torch.load(path_drive_pretrained, map_location=torch.device('cpu')))
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
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), path_model_more_trained)
            # revert model to training mode
            model.train()


    train_bool = False
    if train_bool :
        trainer.set_callback('on_batch_end', batch_end_callback)
        # run the optimization
        trainer.run()

    test_bool = False
    if test_bool:
        list_top = [2, 3, 5, 10]
        path_all_results_texts_models = "/content/drive/MyDrive/minGPT_results/all_conditional_accs.csv"
        with open(path_all_results_texts_models, 'a') as acc_for_context :
          acc_for_context.write("%s,%s,%s,%s,%s,%s,%s,%s\n" % ("model_name", 
          "text_name", "size_content", "sum_correct_pred","top_2",  "top_3", "top5", "top10"))
        #for path in [path_short_training, path_long_training]:
        #   with open(path_save_results + '_acc_for_context.csv', 'w') as acc_for_context :
        #       acc_for_context.write("%s,%s,%s\n" % ("size_content", "sum_correct_pred", "sum_approx_correct_pred"))
        for name_short, name_long in zip(["cable", "easy_money", "willow", "lw1"],
        ["cable_spool_fort_ph_punct.txt", "easy_money_ph_punct.txt",
         "the_black_willow_ph_punct.txt", "lw1_ph_punct.txt"]):
            print(name_short, name_long)
            path_save_results = "/content/drive/MyDrive/minGPT_results/" + model_title + "/"+ name_short +"_more_training"
            # path_save_results = "/content/minGPT4phonemics/results_context"
            # ADD for loop for all 4 files
            # path_save_results = "/content/drive/MyDrive/minGPT_results/" + model_title + "/gpt2_willow"
            text_results_acc_for_context = open(path_save_results + "_acc_for_context" + ".txt", "w")
            text_results_acc_for_context.write(
                "size_context" + "   " + "sum_correct_pred =" + "   " + "sum_approx_correct_pred" + "\n")
            baseline_not_trained = True
            if not baseline_not_trained:
                if on_GPU :
                    model.load_state_dict(torch.load(path_model_more_trained))
                else :
                    model.load_state_dict(torch.load(path_model_more_trained, map_location=torch.device('cpu')))
            with open(path_all_results_texts_models, 'a') as acc_for_context :
                for size_context in [100]: #[1, 2, 3, 5, 10, 20, 30, 40, 50, 100, 500, 1000]:
                    print("size of context =", size_context)
                    sum_correct_pred = 0
                    sum_approx_correct_pred = 0
                    dic_sum_pred_top = {}
                    for top in list_top:
                        dic_sum_pred_top["sum_pred_in_top_"+ str(top)] = 0
                    text_test = open('/content/minGPT4phonemics/bids_anonym_stimuli_text/' + name_long,
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
                    with open(path_save_results_context + '.csv', 'w') as context_csv :
                        context_csv.write("%s,%s,%s,%s\n" % ("ORDER", "target CHARACTER", "RANK", "PROBABILITY"))
                        for pred_char in range(n_char2predict) :
                            context = text_test[pred_char:size_context + pred_char]
                            test_torch = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[
                                None, ...].to(trainer.device)
                            results = model.generate4testing(test_torch, 1, temperature=1.0, do_sample=False, top_k=None,
                                                     return_proba=True, add_layer = add_layer)
                            result_char, results_probas, ordered_ranks = results[0], \
                            list(list(results[1].values())[0][0][0]), list(list(results[1].values())[0][1][0])
                            #print("result_char =", result_char)
                            #print("results_probas =", results_probas)
                            #print("ordered_ranks =", ordered_ranks)
                            char_result = train_dataset.itos[result_char] if result_char < len(train_dataset.itos) else "_"
                            text_results.write(char_result)
                            sum_correct_pred += (char_result == text_test[size_context + pred_char])
                            for (enum_rank, rank0), (enum_proba, proba0) in zip(enumerate(ordered_ranks), enumerate(results_probas)):
                                #print("enum_rank, enum_proba ", enum_rank, enum_proba)
                                rank = rank0.item()
                                proba = proba0.item()
                                #print(rank, train_dataset.stoi[text_test[size_context + pred_char]])
                                rank_of_target = (rank == train_dataset.stoi[text_test[size_context + pred_char]])
                                if rank_of_target:
                                    #print(rank, train_dataset.stoi[text_test[size_context + pred_char]], rank_of_target)
                                    n_rank_target, rank_proba = enum_rank, proba
                                    for top in list_top:
                                        if enum_rank < top:
                                            dic_sum_pred_top["sum_pred_in_top_" + str(top)] += 1
                                    break
                            #range_approx = 3
                            #for approx in range(pred_char - range_approx, pred_char + range_approx) :
                                #if size_context + approx < len(text_test) :
                                    #if (char_result == text_test[size_context + approx]) :
                                        #sum_approx_correct_pred += 1
                            context_csv.write("%s,%s,%s,%s\n" % (pred_char, train_dataset.stoi[text_test[size_context + pred_char]],
                                                       n_rank_target, rank_proba))
                            # if pred_char != 0 and pred_char%1000 == 0:
                            # print("pred_char =", pred_char)
                            # print("sum_correct_pred =", np.round(sum_correct_pred/pred_char, 4))
                            # print("sum_approx_correct_pred =",
                            # np.round(sum_approx_correct_pred/pred_char, 4))
                    sum_correct_pred = np.round(sum_correct_pred / n_char2predict, 4)
                    sum_approx_correct_pred = np.round(sum_approx_correct_pred / n_char2predict, 4)
                    for top in list_top:
                        dic_sum_pred_top["sum_pred_in_top_"+ str(top)] = np.round(dic_sum_pred_top["sum_pred_in_top_"+ str(top)]/n_char2predict, 4)
                    text_results_acc_for_context.write(str(size_context) + "   "
                                                       + str(sum_correct_pred) + "   " + str(
                        sum_approx_correct_pred) + "\n")
                    acc_for_context.write("%s,%s,%s,%s,%s,%s,%s,%s\n" % 
                    (model_after_training, name_short, size_context, sum_correct_pred,
                    dic_sum_pred_top["sum_pred_in_top_"+ str(2)], dic_sum_pred_top["sum_pred_in_top_"+ str(3)],
                    dic_sum_pred_top["sum_pred_in_top_"+ str(5)], dic_sum_pred_top["sum_pred_in_top_"+ str(10)]))
                    text_results.close()
                    # text_results_acc_for_context.flush()
            text_results_acc_for_context.close()

    generate_bool = True
    if generate_bool:
        contexts = [
            "ə lɒŋ taɪm əˈɡəʊ, ɪn ə ɡ",
            "aɪ hæv ə dr",
            "wɛl dʌn ɪz "
        ]
        for context in contexts:
            x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None, ...].to(
                trainer.device)
            y, dic_proba = model.generate4testing(x, 500, temperature=1.0, do_sample=True, top_k=10, add_layer=add_layer)[0]
            df_proba = pd.DataFrame.from_dict(dic_proba)
            completion = ''.join([train_dataset.itos[int(i)] for i in y])
            print(context)
            print(completion)
            print(df_proba)

