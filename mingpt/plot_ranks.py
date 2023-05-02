import os
import sys
import csv
import numpy as np
from pandas import DataFrame as df
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import mne

report_freq = mne.Report()

def plot(df_, x_label, y_label, two_models = True):
    fig, ax = plt.subplots(1, figsize=[26, 6])
    if two_models:
        sns.lineplot(x=x_label, y=y_label, data=df_, hue="model_name", errorbar=('ci', 95))
    else:
        sns.lineplot(x=x_label, y=y_label, data=df_, errorbar=('ci', 95))
    #ax.axhline(0, color="k")
    return fig

df_all_models = pd.read_csv("all_conditional_accs.csv")
df_all_models = df_all_models[df_all_models["size_content"] == str(100)]
df_all_models = df_all_models[df_all_models["text_name"] == "willow"]
print(df_all_models)

list_model = ["gpt-mini", "GPT_mini_more_pretrained", "GPT_mini_untrained", "GPT_mini_trained_200_it"]
list_top = ["top_2","top_3","top5","top10"]
fig, ax = plt.subplots(1, figsize=[26, 6])
for model in list_model:
    y_label = []
    df_model = df_all_models[df_all_models["model_name"] == model]
    for top in list_top:
        print(df_model)
        print(df_model[top].item())
        y_label.append(float(df_model[top].item()))
        print(y_label)
    sns.lineplot(x= list_top, y=y_label, legend='brief', label=model) #hue="model_name", errorbar=('ci', 95)
report_freq.add_figure(fig, 'top ranks along the categories', tags="willow")
report_freq.save("df_top ranks along the categories.html", open_browser=False, overwrite=True)