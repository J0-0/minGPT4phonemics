import os
import sys
import csv
import numpy as np
from pandas import DataFrame as df
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import mne

def plot(df_, x_label, y_label, two_models = True):
    fig, ax = plt.subplots(1, figsize=[26, 6])
    if two_models:
        sns.lineplot(x=x_label, y=y_label, data=df_, hue="coherence", errorbar=('ci', 95))
    else:
        sns.lineplot(x=x_label, y=y_label, data=df_, errorbar=('ci', 95))
    #ax.axhline(0, color="k")
    return fig

report_freq = mne.Report()
dico_its = {0: '\n', 1: ' ', 2: '!', 3: '"', 4: '(', 5: ')', 6: ',', 7: '.', 8: '1', 9: ':', 10: ';', 11: '?', 12: '[', 13: '\\', 14: ']', 15: 'a', 16: 'b', 17: 'c', 18: 'd', 19: 'e', 20: 'f', 21: 'h', 22: 'i', 23: 'j', 24: 'k', 25: 'l', 26: 'm', 27: 'n', 28: 'o', 29: 'p', 30: 'q', 31: 'r', 32: 's', 33: 't', 34: 'u', 35: 'v', 36: 'w', 37: 'x', 38: 'y', 39: 'z', 40: '|', 41: '¡', 42: '«', 43: '»', 44: '¿', 45: 'æ', 46: 'ç', 47: 'ð', 48: 'ø', 49: 'ŋ', 50: 'ɐ', 51: 'ɑ', 52: 'ɔ', 53: 'ɕ', 54: 'ɖ', 55: 'ə', 56: 'ɚ', 57: 'ɛ', 58: 'ɜ', 59: 'ɡ', 60: 'ɣ', 61: 'ɪ', 62: 'ɫ', 63: 'ɬ', 64: 'ɭ', 65: 'ɲ', 66: 'ɹ', 67: 'ɾ', 68: 'ʀ', 69: 'ʁ', 70: 'ʂ', 71: 'ʃ', 72: 'ʈ', 73: 'ʉ', 74: 'ʊ', 75: 'ʋ', 76: 'ʌ', 77: 'ʑ', 78: 'ʒ', 79: 'ʔ', 80: 'ʝ', 81: 'ʰ', 82: 'ʲ', 83: 'ː', 84: '̃', 85: '̩', 86: 'β', 87: 'θ', 88: 'ᵐ', 89: 'ᵻ', 90: '—', 91: '“', 92: '”', 93: '…'}
list_order = []
list_p_order = []
np_order = np.zeros(15)
np_cat = np.zeros(94)
list_cat = []
list_ph_cat = []
list_p_cat = []
list_model = []
for path, model in zip(["gpt-mini-results/willow_100.csv", "GPT2-medium/willow_100.csv", "GPT2/willow_100.csv"],
                       ["GPT-mini trained from scratch", "GPT2-medium-fine-tuned", "GPT2-fine-tuned"]):
    df_freq0 = pd.read_csv(path)
    df_freq = df_freq0[df_freq0["predicted CHARACTER"] == df_freq0["target CHARACTER"]] #[11093 rows x 4 columns]
    df_missed_freq = df_freq0[df_freq0["predicted CHARACTER"] != df_freq0["target CHARACTER"]] #[10715 rows x 4 columns]
    # SUCCESS  0.8435289191381952 MISSED  0.4102891180587961
    print(model)
    print("SUCCESS ", df_freq.shape[0], np.nanmean(df_freq["PROBABILITY"]), "MISSED ", df_missed_freq.shape[0], np.nanmean(df_missed_freq["PROBABILITY"]))
    fig_df_freq = plot(df_ = df_freq, x_label = "ORDER", y_label ="PROBABILITY", two_models = False)
    report_freq.add_figure(fig_df_freq, 'frequencies along the text for '+model, tags="willow")
    order_after_space = 0
    categories = {}
    dic_order_freq = {}
    dic_order_quantity = {}
    dic_order_cat = {}

    for index, row in df_freq0.iterrows():
        el = row["target CHARACTER"]
        int_el = int(el)
        if el == 1:
            order_after_space = 0
        else:
            order_after_space += 1
        if order_after_space not in dic_order_quantity.keys():
            dic_order_quantity[order_after_space] = 0
        dic_order_quantity[order_after_space] += 1
        if int_el not in dic_order_cat.keys():
            dic_order_cat[int_el] = 0
        dic_order_cat[int_el] += 1
        if row["predicted CHARACTER"] == row["target CHARACTER"]:
            list_order.append(order_after_space)
            np_order[order_after_space] += 1
            np_cat[int_el] += 1
            list_p_order.append(row["PROBABILITY"])
            list_cat.append(row["predicted CHARACTER"])
            list_p_cat.append(row["PROBABILITY"])
            list_ph_cat.append(dico_its[row["predicted CHARACTER"]])
            list_model.append(model)
            if order_after_space not in dic_order_freq.keys():
                dic_order_freq[order_after_space] = []
            dic_order_freq[order_after_space].append(row["PROBABILITY"])
            ph_el = dico_its[el]
            if ph_el not in categories.keys():
                categories[ph_el] = []
            categories[ph_el].append(row["PROBABILITY"])
    dic_order_freq_mean, categories_mean = {}, {}
    for key in dic_order_freq.keys():
        dic_order_freq_mean[key] = np.round(np.nanmean(dic_order_freq[key]), 4)
        dic_order_freq[key] = np.array(dic_order_freq[key])
    for key in categories.keys():
        categories_mean[key] = np.round(np.nanmean(categories[key]), 4)

    for n, el in enumerate(np_order):
        np_order[n] = np.round(np_order[n]/dic_order_quantity[n], 4)

    #print(dic_order_cat)
    #for n, el in enumerate(np_cat):
        #np_cat[n] = np.round(np_cat[n]/dic_order_cat[n], 4)

pd_order_freq = df()
pd_order_freq["order in sentence"] = list_order
pd_order_freq["probabilities"] = list_p_order
pd_order_freq["coherence"] = list_model

pd_order_sum = df()
pd_order_sum["order in sentence"] = np.arange(15)
pd_order_sum["proportion success in order"] = np_order

#pd_cat_sum = df()
#pd_cat_sum["categories"] = np.arange(94)
#pd_cat_sum["proportion success in categories"] = np_cat

pd_cat_freq = df()
pd_cat_freq["categories"] = list_ph_cat
pd_cat_freq["probabilities"] = list_p_cat
pd_cat_freq["coherence"] = list_model

fig_freq_proba = plot(df_ = pd_order_freq, x_label = "order in sentence", y_label ="probabilities")
report_freq.add_figure(fig_freq_proba, 'frequencies along the sentence', tags="willow")

fig_freq_order = plot(df_ = pd_order_sum, x_label = "order in sentence", y_label ="proportion success in order", two_models = False)
report_freq.add_figure(fig_freq_order, 'proportion of successful predictions along the sentence', tags="willow")

fig_cat_proba = plot(df_ = pd_cat_freq, x_label = "categories", y_label ="probabilities")
report_freq.add_figure(fig_cat_proba, 'frequencies along the categories', tags="willow")

#fig_freq_cat = plot(df_ = pd_cat_freq, x_label = "categories", y_label ="proportion success in categories", two_models = False)
#report_freq.add_figure(fig_freq_cat, 'proportion of successful predictions along the categories', tags="willow")

report_freq.save("df_freq_order_willow.html", open_browser=False, overwrite=True)





