import csv
import re
import select
from tkinter import font
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import scienceplots

from scipy.signal import savgol_filter
from pathvalidate import sanitize_filename
from scipy.ndimage import gaussian_filter1d

import numpy as np

import copy
import logging


# global abc
abc = 0
full_abc = ['a', 'b', 'c', 'd', 'e', 'f',
            'g', 'h', 'i', 'j', 'k', 'l', 'm']

def _fix_legend(all_axes, fig, bbox_to_anchor=(0.5, 0.3)):
    all_axes = all_axes.flatten()
    for ax in all_axes:
        try:
            ax.get_legend().remove()
        except:
            pass

    # Create a common legend
    temp_axis = all_axes[0]
    handles, labels = temp_axis.get_legend_handles_labels()

    # Add the legend below the subplots
    fig.legend(handles, labels, loc='lower center',
               ncol=2, bbox_to_anchor=bbox_to_anchor, handletextpad=0.01,
               columnspacing=0.1,
               borderpad=0.1,
               labelspacing=0.1,)


def _get_paper_name(name):

    d = {}

    d['openai-community/openai-gpt'] = 'GPT'
    d['openai-communityopenai-gpt'] = 'GPT'
    d['google-bert/bert-base-cased'] = 'BERT'
    d['resnet18'] = 'ResNet'
    d['densenet121'] = 'DenseNet'

    d['dbpedia_14'] = 'DBpedia'
    d['yahoo_answers_topics'] = 'Yahoo-Answers'
    d['mnist'] = 'MNIST'
    d['cifar10'] = 'CIFAR10'
    d['pathmnist'] = 'Colon-Pathology'
    d['organamnist'] = 'Abdominal-CT'
    d['PathologicalPartitioner-3'] = 'Pathological'
    d['non_iid_dirichlet'] = 'Dirichlet'

    return d[name]

def smooting_filter(column_values):
    # return savgol_filter(column_values, window_length=6, polyorder=1)
    # return moving_average(column_values, window_size=4)
    return gaussian_filter1d(column_values, sigma=2)

import hashlib

def get_hashed_name(name, algorithm='md5'):
    """
    Returns the hash of the given name using the specified algorithm.

    :param name: The string to be hashed.
    :param algorithm: The hashing algorithm to use ('md5', 'sha1', 'sha256').
    :return: The hashed name as a hexadecimal string.
    """
    # Choose the hashing algorithm
    if algorithm == 'md5':
        hash_object = hashlib.md5(name.encode())
    elif algorithm == 'sha1':
        hash_object = hashlib.sha1(name.encode())
    elif algorithm == 'sha256':
        hash_object = hashlib.sha256(name.encode())
    else:
        raise ValueError("Unsupported algorithm. Use 'md5', 'sha1', or 'sha256'.")
    
    # Generate the hash
    hashed_name = hash_object.hexdigest()
    return hashed_name

def convert_cache_to_csv(cache):
    keys = cache.keys()
    csv_paths = []
    for key in keys:
        # logging.info(f"Plotting for key: {key}")
        round2prov_result = cache[key]["round2prov_result"]
        prov_cfg = cache[key]["prov_cfg"]
        avg_prov_time_per_round = cache[key]["avg_prov_time_per_round"]

        each_round_prov_result = []

        for r2prov in round2prov_result:
            r2prov['training_cache_path'] = cache[key]["training_cache_path"]
            r2prov['avg_prov_time_per_round'] = avg_prov_time_per_round
            r2prov['prov_cfg'] = prov_cfg
            r2prov['Model'] = prov_cfg.model.name
            r2prov['Dataset'] = prov_cfg.dataset.name
            r2prov['Num Clients'] = prov_cfg.num_clients
            r2prov['Dirichlet Alpha'] = prov_cfg.dirichlet_alpha

            if 'Error' in r2prov:
                continue

            for m, v in r2prov['eval_metrics'].items():
                r2prov[m] = v
            # if 'avg_prov_time_per_input' in r2prov:
            #     r2prov['abc_avg_prov_time_per_input'] = r2prov['avg_prov_time_per_input']
            #     logging.info(f"avg_prov_time_per_input: {r2prov['avg_prov_time_per_input']}")
            each_round_prov_result.append(copy.deepcopy(r2prov))

        df = pd.DataFrame(each_round_prov_result)
        key = sanitize_filename(key)
        csv_path = f"results_csvs/prov_{key}.csv"

        if len(csv_path) > 250:
            logging.warn(f"CSV path too long, using hashed name: {csv_path}")
            hashed_name = get_hashed_name(csv_path)
            csv_path = f"results_csvs/{hashed_name}.csv"
            logging.warn(f"hashed name: {csv_path}")



        csv_paths.append(csv_path)
        df.to_csv(csv_path)

    with open('csv_paths.log', 'w') as f:
        for path in sorted(csv_paths):
            f.write(f"{path}\n")


def line_plot(axis, x, y, label, linestyle, linewidth=2):
    axis.plot(x, y, label=label, linestyle=linestyle, linewidth=linewidth)


#     plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=.1)
#     _save_plot(fig, "text_image_audio_classification_results")

def _save_plot(fig, filename_base):
    for ext in ['png', 'svg', 'pdf']:
        plt.savefig(f"graphs/{ext}s/{filename_base}.{ext}",
                    bbox_inches='tight', format=ext, dpi=600)

    plt.close('all')


def _call_before_everyPlot(width_inches=3.3374, height_inches=3.3374/1.618, nrows=1, ncols=1, **kwargs):
    # mpl.use("pgf")
    # sns.set_theme(
    #     context="paper", style="ticks", palette="colorblind", font_scale=1, font="serif"
    # )

    plt.style.use(['science', 'ieee', 'grid', 'no-latex'])
    fig, axes = None, None
    if nrows != -1 and ncols != -1:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(
            width_inches, height_inches), **kwargs)
    else:
        fig, axes = plt.subplots(
            figsize=(width_inches, height_inches), **kwargs)

    return fig, axes


def _plot_label_distribution_motivation():
    # df = pd.read_csv(
    #     'results_csvs/label_distribution_proper_labels_without_flip_pathmnist.csv', index_col=0)
    # print(df)

    # fig, ax = _call_before_everyPlot(
    #     width_inches=3.3374*1.6, height_inches=3.3374/1.618, nrows=1, ncols=1)
    # sns.heatmap(df, annot=True, fmt="d",  cmap='Blues',  linewidths=.5)

    # plt.xlabel('Hospital ID')
    # plt.ylabel('Medical Imaging Label')
    # plt.tight_layout()
    # _save_plot(fig, f"label_distribution_proper_labels_without_flip_pathmnist")

    

    def plot_partition_labels_distribution():
        # Dataset provided by the user
        data = {
            'Labels': ['Adipose', 'Background', 'Debris', 'Lymphocytes', 'Mucus', 'Smooth Muscle', 
                    'Normal Colon Mucosa', 'Cancer-associated Stroma', 'Colorectal Adenocarcinoma'],
            'H0': [670, 667, 711, 0, 0, 0, 0, 0, 0],
            'H1': [0, 568, 623, 857, 0, 0, 0, 0, 0],
            'H2': [0, 0, 602, 807, 639, 0, 0, 0, 0],
            'H3': [0, 0, 0, 737, 508, 803, 0, 0, 0],
            'H4': [0, 0, 0, 0, 587, 846, 615, 0, 0],
            'H5': [0, 0, 0, 0, 0, 891, 533, 624, 0],
            'H6': [0, 0, 0, 0, 0, 0, 528, 635, 885],
            'H7': [516, 0, 0, 0, 0, 0, 0, 611, 921],
            'H8': [540, 521, 0, 0, 0, 0, 0, 0, 987],
            'H9': [672, 666, 710, 0, 0, 0, 0, 0, 0]
        }

        # Convert the data into a DataFrame
        df = pd.DataFrame(data)

        # Transform the DataFrame for easier plotting
        df_melted = df.melt(id_vars='Labels', var_name='Partition ID', value_name='Count')

        fig, axes =  _call_before_everyPlot(width_inches=3.3374*4.6, height_inches=(3.3374*2)/1.618, nrows=1, ncols=1)
         
        # Summing counts for stacked bars
        df_melted_pivot = df_melted.pivot_table(index='Partition ID', columns='Labels', values='Count', aggfunc='sum', fill_value=0)

        # Stacked bar plot
        df_melted_pivot.plot(kind='bar', stacked=True)

        plt.title('Per Hospital Labels Distribution')
        plt.xlabel('Hospital ID')
        plt.ylabel('Number of Data Points')
        # plt.legend(title='Labels', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.legend(title='Labels', bbox_to_anchor=(0.5, 1.15), loc='center', ncol=3, frameon=False)
    
        plt.tight_layout()
        _save_plot(fig, f"label_distribution_proper_labels_without_flip_pathmnist")
        # plt.show()

    # To use the function, simply call it
    plot_partition_labels_distribution()




def _plot_text_image_audio_classification_results():
    # from scipy.interpolate import UnivariateSplin
    def _getdf(mname, dname, alpha, trounds):
        fname = f'results_csvs/prov_For-Main-Training--{mname}-{dname}-faulty_clients[[]]-noise_rateNone-TClients100-fedavg-(R{trounds}-clientsPerR10)-non_iid_dirichlet{alpha}-batch32-epochs2-lr0.001.csv'
        return pd.read_csv(fname)

    def _plot(ax, mname, dname, alpha, trounds):
        df = _getdf(mname, dname, alpha, trounds)
        model = _get_paper_name(df['Model'][0])
        dataset = _get_paper_name(df['Dataset'][0])

        # # Apply Savitzky-Golay filter
        df['Accuracy'] = df['Accuracy'] * 100
        df['test_data_acc'] = df['test_data_acc'] * 100

        ax.plot(range(len(df)), smooting_filter(
            df['Accuracy']), label='TraceFL-Smooth')
        ax.plot(range(len(df)), smooting_filter(
            df['test_data_acc']), label='FL Training-Smooth')
        temp_dict = {'Average Accuracy': df['Accuracy'].mean(
        ), 'Total Rounds': trounds, 'Total Accuracy': sum(df['Accuracy'])}

        global abc
        title = (f"{full_abc[abc]}) {dataset}\n{model}")
        abc += 1
        ax.text(0.5, 0.5, f"TraceFL \n Avg. Acc {temp_dict['Average Accuracy']:.1f}",
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=9)
        ax.set_title(title)
        ax.legend()
        all_config_summary.append(temp_dict)
        # return temp_dict

    def _plot_model_dataset_config(axes, mnames, dnames, num_rounds):
        _plot(axes[0], mnames[0], dnames[0], alpha, num_rounds)
        _plot(axes[1], mnames[0], dnames[1], alpha, num_rounds)
        _plot(axes[2], mnames[1], dnames[0], alpha, num_rounds)
        _plot(axes[3], mnames[1], dnames[1], alpha, num_rounds)

    # for alpha in [0.1, 0.2, 0.3]:
    alpha = 0.3
    global abc
    abc = 0
    full_abc = ['a', 'b', 'c', 'd', 'e', 'f',
                'g', 'h', 'i', 'j', 'k', 'l', 'm']
    width = 3.3374*1.6
    height = 3.3374 * 1.5
    fig, all_axes = _call_before_everyPlot(
        width_inches=width, height_inches=height, nrows=3, ncols=4, sharey=True)

    all_config_summary = []

    text_models = ['openai-communityopenai-gpt',
                    'google-bertbert-base-cased']
    text_datasets = ['dbpedia_14', 'yahoo_answers_topics']

    image_models = ['resnet18', 'densenet121']
    standd_datasets = ['mnist', 'cifar10']
    medical_datasets = ['pathmnist', 'organamnist']

    _plot_model_dataset_config(
        all_axes[0], image_models, medical_datasets, 25)
    _plot_model_dataset_config(
        all_axes[1], text_models, text_datasets, num_rounds=25)
    _plot_model_dataset_config(
        all_axes[2], image_models, standd_datasets, 50)

    
    _fix_legend(all_axes, fig, bbox_to_anchor=(0.5, 0.04))

    fig.supxlabel('Communication Rounds', fontsize=12)
    fig.supylabel('Accuracy (%)', fontsize=12, )

    total_rounds = sum([x['Total Rounds'] for x in all_config_summary])
    total_accuracy = sum([x['Total Accuracy'] for x in all_config_summary])
    average_accuracy = total_accuracy / total_rounds
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    plt.tight_layout()
    fname = f"text_image_audio_classification_results_{alpha}_alpha" 
    _save_plot(fig, fname)
    
    logging.info(f"-------------- {fname} --------------")  
    logging.info(f"Total Rounds: {total_rounds}")
    logging.info(f"Total Accuracy: {total_accuracy}")
    logging.info(f"Average Accuracy: {average_accuracy}")
    logging.info(f'Total Models Trained  {total_rounds * 10}')

    # return the above metrics in a dict
    return {'Total Rounds': total_rounds, 'Total Accuracy': total_accuracy, 'Average Accuracy': average_accuracy, 'Total Models Trained': total_rounds * 10}




def _table_and_graph_scalability_results():
    def _get_df_scaling_clients(num_clients):
        fname = f'results_csvs/prov_Scaling-openai-communityopenai-gpt-dbpedia_14-faulty_clients[[]]-noise_rateNone-TClients{num_clients}-fedavg-(R15-clientsPerR10)-non_iid_dirichlet0.3-batch32-epochs2-lr0.001.csv'
        return pd.read_csv(fname)

    def _get_df_clients_per_round(clients_per_round):
        fname = f'results_csvs/prov_Scaling-openai-communityopenai-gpt-dbpedia_14-faulty_clients[[]]-noise_rateNone-TClients400-fedavg-(R15-clientsPerR{clients_per_round})-non_iid_dirichlet0.3-batch32-epochs2-lr0.001.csv'
        return pd.read_csv(fname)

    def _get_df_num_rounds():
        fname = f'results_csvs/prov_Scaling-openai-communityopenai-gpt-dbpedia_14-faulty_clients[[]]-noise_rateNone-TClients400-fedavg-(R100-clientsPerR10)-non_iid_dirichlet0.3-batch32-epochs2-lr0.001.csv'
        return pd.read_csv(fname)

    def _plot(ax, df, r):
        df['Accuracy'] = df['Accuracy'][:r] * 100
        df['test_data_acc'] = df['test_data_acc'][:r] * 100
        ax.plot(range(len(df)), smooting_filter(
            df['Accuracy']), label='TraceFL-Smooth')
        ax.plot(range(len(df)), smooting_filter(
            df['test_data_acc']), label='FL Training-Smooth')
        # ax.set_title(title)
        # set y limit
        ax.set_ylim([0, 105])

        tracefl_avg_acc = df['Accuracy'].mean()

        ax.text(0.5, 0.5, f"TraceFL \n Avg. Acc {tracefl_avg_acc:.1f} %",
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=9)

        # set x label
        ax.set_xlabel("Communication Rounds")
        ax.set_ylabel("Accuracy (%)")

    # def smooting_filter(column_values):
    #     return gaussian_filter1d(column_values, sigma=2)

    scaling_clients = [200, 400, 600, 800, 1000]
    per_round_clients = [20, 30, 40, 50]

    scaling_total_clients_dicts = []
    total_rounds_cleints200_1000 = 0 
    total_accuracy_clients200_1000 = 0
    for num_clients in scaling_clients:
        df = _get_df_scaling_clients(num_clients)
        total_rounds_cleints200_1000 += len(df)
        temp_dict = {}
        temp_dict['Total Clients'] = num_clients
        temp_dict['Global Model Accuracy \%'] = round(
            df['test_data_acc'].max()*100, 2)
        temp_dict['TraceFL Avg. Accuracy \%'] = round(
            df['Accuracy'].mean()*100, 2)
        scaling_total_clients_dicts.append(temp_dict)
        total_accuracy_clients200_1000 += (df['Accuracy'].sum()*100)

    total_model_trained_clients200_1000 = total_rounds_cleints200_1000 * 10
    # log the above metrics
    logging.info(f"-------------- Total Clients 200-1000 --------------")
    logging.info(f"Total Rounds: {total_rounds_cleints200_1000}")
    logging.info(f"Total Accuracy: {total_accuracy_clients200_1000}")
    logging.info(f"Average Accuracy: {total_accuracy_clients200_1000/total_rounds_cleints200_1000}")
    logging.info(f'Total Models Trained:  {total_model_trained_clients200_1000}')


    df_toal_clients = pd.DataFrame(scaling_total_clients_dicts)
    
    logging.info(df_toal_clients) 
    # Convert DataFrame to LaTeX with float values formatted to 2 decimal points
    latex_code_toal_clients = df_toal_clients.to_latex(
        index=False, float_format="%.2f")

    # graphs/tables
    with open("graphs/tables/scalability_results_table_total_clients.tex", "w") as f:
        f.write(latex_code_toal_clients)

    # clients per round
    clients_per_round_dicts = []
    total_rounds_per_round_client20_50 = 0
    total_accuracy_per_round_client20_50 = 0
    total_model_trained_per_round_client20_50 = 0
    for clients_per_round in per_round_clients:
        df = _get_df_clients_per_round(clients_per_round)
        temp_dict = {}
        temp_dict['Clients Per Round'] = clients_per_round
        temp_dict['Global Model Accuracy \%'] = round(
            df['test_data_acc'].max()*100, 2)
        temp_dict['TraceFL Avg. Accuracy \%'] = round(
            df['Accuracy'].mean()*100, 2)
        clients_per_round_dicts.append(temp_dict)
        total_rounds_per_round_client20_50 += len(df)
        total_accuracy_per_round_client20_50 += (df['Accuracy'].sum()*100)
        total_model_trained_per_round_client20_50 += len(df) * clients_per_round


    # log the above metrics
    logging.info(f"-------------- Scalability Clients Per Round 20-50 --------------")
    logging.info(f"Total Rounds: {total_rounds_per_round_client20_50}")
    logging.info(f"Total Accuracy: {total_accuracy_per_round_client20_50}")
    logging.info(f"Average Accuracy: {total_accuracy_per_round_client20_50/total_rounds_per_round_client20_50}")
    logging.info(f'Total Models Trained:  {total_model_trained_per_round_client20_50}')


    df_clients_per_round = pd.DataFrame(clients_per_round_dicts)
    logging.info(df_clients_per_round)

    # Convert DataFrame to LaTeX with float values formatted to 2 decimal points

    latex_code_clients_per_round = df_clients_per_round.to_latex(
        index=False, float_format="%.2f")
    with open("graphs/tables/scalability_results_table_clients_per_round.tex", "w") as f:
        f.write(latex_code_clients_per_round)

    width = 3.3374
    height = 3.3374/1.6
    fig, all_axes = _call_before_everyPlot(
        width_inches=width, height_inches=height, nrows=1, ncols=1, sharey=True)

    # Plot for varying number of rounds
    logging.info(f"-------------- Scalability Number of Rounds 80 --------------")
    
    total_rounds_num_rounds_80 = 80
    total_model_trained_rounds_80 = 80 * 10
    num_rounds_exp = 80
    df = _get_df_num_rounds()
    _plot(all_axes, df, num_rounds_exp)
    total_accuracy_num_rounds_80 = (df['Accuracy'][:num_rounds_exp].sum())
    logging.info(f"Total Rounds: {num_rounds_exp}")
    logging.info(f"Total Accuracy: {total_accuracy_num_rounds_80}")
    logging.info(f"Average Accuracy: {total_accuracy_num_rounds_80/num_rounds_exp}")
    logging.info(f'Total Models Trained:  {total_model_trained_rounds_80}')


    # Add the legend below the subplots
    all_axes.legend(loc='lower center',
                    ncol=2, bbox_to_anchor=(0.5, 0.05), handletextpad=0.01,
                    columnspacing=0.1, borderpad=0.1, labelspacing=0.1)

    plt.tight_layout()
    _save_plot(fig, f"scalability_results_400_clients_rounds_{num_rounds_exp}")

    # return the above metrics in a dict and first sum them all
    total_rounds = total_rounds_cleints200_1000 + total_rounds_per_round_client20_50 + num_rounds_exp
    total_accuracy = total_accuracy_clients200_1000 + total_accuracy_per_round_client20_50 + total_accuracy_num_rounds_80
    average_accuracy = total_accuracy / total_rounds
    total_model_trained = total_model_trained_clients200_1000 + total_model_trained_per_round_client20_50 + total_model_trained_rounds_80

    return {'Total Rounds': total_rounds, 'Total Accuracy': total_accuracy, 'Average Accuracy': average_accuracy, 'Total Models Trained': total_model_trained}
     




def _plot_differential_privacy_results():
    def _get_avg_prov_and_max_acc_differential_privacy(dclip, alpha):
        all_dfs = [_get_df_gpt(noise=ds, clip=dclip, alpha=alpha)
                   for ds in dp_noises]
        average_prov_on_each_alpha = [
            df['Accuracy'].mean()*100 for df in all_dfs]
        max_gm_acc_on_each_alpha = [
            df['test_data_acc'].max()*100 for df in all_dfs]
        return {'prov': average_prov_on_each_alpha, 'gm': max_gm_acc_on_each_alpha, 'Total Rounds': sum([len(df) for df in all_dfs]), 'Total Accuracy': sum([df['Accuracy'].sum()*100 for df in all_dfs])}

    def _get_df_gpt(noise, clip, alpha):
        fname = f'results_csvs/prov_DP-(noise{noise}+clip{clip})-DP-text-openai-communityopenai-gpt-dbpedia_14-faulty_clients[[]]-noise_rateNone-TClients100-fedavg-(R15-clientsPerR10)-non_iid_dirichlet{alpha}-batch32-epochs2-lr0.001.csv'
        return pd.read_csv(fname)

    def _plot(ax, clip, alpha):
        temp_dict = _get_avg_prov_and_max_acc_differential_privacy(clip, alpha)
        ax.plot(dp_noises, temp_dict['prov'], label='Avg. TraceFL Accuracy')
        ax.plot(dp_noises, temp_dict['gm'], label='FL Training Accuracy')
        # ax.set_title(f"Norm Clip {clip}")
        ax.set_xlabel("Differential Privacy Noise")
        # ax.set_ylabel("Accuracy (%)")
        ax.legend()
        return temp_dict

    dp_noises = [0.0001, 0.0003, 0.0007, 0.0009, 0.001, 0.003]
    # dp_clip = [15 50]
    alpha_for_dp_exp = 0.2

    fig, all_axes = _call_before_everyPlot(
        width_inches=3.3374*1.3, height_inches=3.3374/1.718, nrows=1, ncols=1, sharey=True)

    temp_dict1 =  _plot(all_axes, 50, alpha_for_dp_exp)
    # _plot(all_axes[1], dp_clip[1], alpha_for_dp_exp)

    fig.supylabel('Accuracy (%)')

    all_axes = all_axes

    # for ax in all_axes:
    #     try:
    #         ax.get_legend().remove()
    #     except:
    #         pass

    all_axes.get_legend().remove()

    # Create a common legend
    temp_axis = all_axes
    handles, labels = temp_axis.get_legend_handles_labels()

    # Add the legend below the subplots
    fig.legend(handles, labels, loc='lower center',
               ncol=2, bbox_to_anchor=(0.5, 0.3),    handletextpad=0.01,
               columnspacing=0.1,
               borderpad=0.1,
               labelspacing=0.1,)

    plt.tight_layout()
    _save_plot(fig, f"differential_privacy_results_alpha_{alpha_for_dp_exp}")

    # table plotting 
    dp_noises = [0.003]
    temp_dict2 =  _get_avg_prov_and_max_acc_differential_privacy(dclip=15, alpha=0.3)
    dict_for_df2 = {'DP Noise':0.003, 'DP Clip':15, 'FL Training Accuracy': temp_dict2['gm'], 'TraceFL Avg. Accuracy': temp_dict2['prov']}

    dp_noises = [0.006]
    temp_dict3 =  _get_avg_prov_and_max_acc_differential_privacy(dclip=10, alpha=0.3)
    dict_for_df3 = {'DP Noise':0.006, 'DP Clip':10, 'FL Training Accuracy': temp_dict3['gm'], 'TraceFL Avg. Accuracy': temp_dict3['prov']}

    dp_noises = [0.012]
    temp_dict4 =  _get_avg_prov_and_max_acc_differential_privacy(dclip=15, alpha=0.3)
    dict_for_df4 = {'DP Noise':0.012, 'DP Clip':15, 'FL Training Accuracy': temp_dict4['gm'], 'TraceFL Avg. Accuracy': temp_dict4['prov']}


    df = pd.DataFrame([dict_for_df2, dict_for_df3, dict_for_df4])

    logging.info(f'------- DP Results Table -------')
    logging.info(df)

    
    # Convert DataFrame to LaTeX with float values formatted to 2 decimal points
    latex_code = df.to_latex(index=False)
    with open("graphs/tables/differential_privacy_results_table.tex", "w") as f:
        f.write(latex_code)

    # merge the two dicts
    total_rounds = temp_dict1['Total Rounds'] + temp_dict2['Total Rounds'] + temp_dict3['Total Rounds'] + temp_dict4['Total Rounds']
    total_accuracy = temp_dict1['Total Accuracy'] + temp_dict2['Total Accuracy'] + temp_dict3['Total Accuracy'] + temp_dict4['Total Accuracy']
    average_accuracy = total_accuracy / total_rounds
    total_model_trained = total_rounds * 10

    logging.info(f"-------------- differential_privacy_results_alpha_{alpha_for_dp_exp} --------------")
    logging.info(f"Total Rounds: {total_rounds}")
    logging.info(f"Total Accuracy: {total_accuracy}")
    logging.info(f"Average Accuracy: {average_accuracy}")
    logging.info(f'Total Models Trained:  {total_model_trained}')

    return {'Total Rounds': total_rounds, 'Total Accuracy': total_accuracy, 'Average Accuracy': average_accuracy, 'Total Models Trained': total_model_trained}

   



def _plot_dirchlet_alpha_vs_accuracy():
    
    def _getdf(mname, dname, alpha):
        fname = f'results_csvs/prov_Dirichlet-Alpha-{mname}-{dname}-faulty_clients[[]]-noise_rateNone-TClients100-fedavg-(R15-clientsPerR10)-non_iid_dirichlet{alpha}-batch32-epochs2-lr0.001.csv'
        return pd.read_csv(fname)

    
    def _get_avg_prov_and_max_acc(mname, dname):
        all_dfs = [_getdf(mname, dname, a) for a in all_alphas]
        average_prov_on_each_alpha = [
            df['Accuracy'].mean()*100 for df in all_dfs]
        max_gm_acc_on_each_alpha = [
            df['test_data_acc'].max()*100 for df in all_dfs]
        return {'prov': average_prov_on_each_alpha, 'gm': max_gm_acc_on_each_alpha, 'Total Rounds': sum([len(df) for df in all_dfs]), 'Total Accuracy': sum([df['Accuracy'].sum()*100 for df in all_dfs])}

    def _plot(ax, mname, dname):
        global abc
        dataset = _get_paper_name(dname)
        temp_dict = _get_avg_prov_and_max_acc(mname, dname)

        ax.plot(all_alphas, temp_dict['prov'], label='Avg. TraceFL Accuracy')
        ax.plot(all_alphas, temp_dict['gm'], label='Max Global Model Accuracy')

        title = f"{full_abc[abc]}) {dataset}"
        logging.info(f"Dirchelet Title: {title}")
        logging.info(f'alpha {all_alphas[0:5]}, gm {temp_dict["gm"][0:5]}')



        ax.set_title(title)
        ax.legend()
        # set y limit
        # ax.set_ylim([0, 105])
        abc += 1
        return temp_dict
    
    global abc
    abc = 0

    width = 3.3374*1.6
    height = 3.3374 * 1.5
    # width = 3.3374 * 1.6
    # height = 3.3374/1.618
    all_alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    fig, axes = _call_before_everyPlot(
        width_inches=width, height_inches=height, nrows=3, ncols=2, sharey=True)

    temp_dict1 =  _plot(axes[0][0], 'densenet121', 'pathmnist')
    temp_dict2 =  _plot(axes[0][1], 'densenet121', 'organamnist')
    temp_dict3 =  _plot(axes[1][0], 'densenet121', 'mnist')
    temp_dict4 =  _plot(axes[1][1], 'densenet121', 'cifar10')
    temp_dict5 =  _plot(axes[2][0], 'openai-communityopenai-gpt', 'yahoo_answers_topics')
    temp_dict6 =  _plot(axes[2][1], 'openai-communityopenai-gpt', 'dbpedia_14')
    


    total_rounds = temp_dict1['Total Rounds'] + temp_dict2['Total Rounds'] + temp_dict3['Total Rounds'] + temp_dict4['Total Rounds'] + temp_dict5['Total Rounds'] + temp_dict6['Total Rounds']
    total_model_trained = total_rounds * 10 
    total_accuracy = temp_dict1['Total Accuracy'] + temp_dict2['Total Accuracy'] + temp_dict3['Total Accuracy'] + temp_dict4['Total Accuracy'] + temp_dict5['Total Accuracy'] + temp_dict6['Total Accuracy']

    fig.supxlabel('Dirichlet Alpha', fontsize=12)
    fig.supylabel('Accuracy (%)', fontsize=12)
    _fix_legend(axes, fig, bbox_to_anchor=(0.5, 0.04))
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    plt.tight_layout()
    _save_plot(fig, "dirchlet_alpha_vs_accuracy_medical_text")

    logging.info(f"-------------- dirchlet_alpha_vs_accuracy_medical_text --------------")
    logging.info(f"Total Rounds: {total_rounds}")
    logging.info(f"Total Accuracy: {total_accuracy}")
    logging.info(f"Average Accuracy: {total_accuracy/total_rounds}")
    logging.info(f'Total Models Trained:  {total_model_trained}')

    return {'Total Rounds': total_rounds, 'Total Accuracy': total_accuracy, 'Average Accuracy': total_accuracy/total_rounds, 'Total Models Trained': total_model_trained}


# =================================== The Above Plotting Are Finalized ==============================================================
def plot_label_distribution_heatmap_concrete(client2class, labelid2labelname, fname_type):
    # Identifying all labels and sorting client IDs
    all_labels = set()
    for class_data in client2class.values():
        all_labels.update(class_data.keys())
    all_labels = sorted(all_labels)
    client_ids = sorted(client2class.keys(), key=lambda x: int(x[1:]))

    # Preparing DataFrame for seaborn heatmap
    data_matrix = []
    for label in all_labels:
        row = [client2class[client].get(label, 0) for client in client_ids]
        data_matrix.append(row)

    if labelid2labelname is None:
        labelid2labelname = {label: label for label in all_labels}

    df1 = pd.DataFrame(data_matrix, index=[
                       label for label in all_labels], columns=client_ids)
    df2 = pd.DataFrame(data_matrix, index=[
                       labelid2labelname[label] for label in all_labels], columns=client_ids)

    df2.to_csv(
        f'results_csvs/label_distribution_proper_labels_{fname_type}.csv')

    print(f'dataframe: \n{df2}')
    print(f'dataframe: \n{df1}')

    fig, ax = _call_before_everyPlot(
        width_inches=3.3374*1.6, height_inches=3.3374/1.618, nrows=1, ncols=1)
    sns.heatmap(df2, annot=True, fmt="d",  cmap='Blues',  linewidths=.5)

    plt.xlabel('Hospital ID')
    plt.ylabel('Medical Imaging Label')
    plt.tight_layout()
    _save_plot(fig, f"label_distribution_proper_labels_{fname_type}")

    fig, ax = _call_before_everyPlot(
        width_inches=3.3374*1.6, height_inches=3.3374/1.618, nrows=1, ncols=1)
    sns.heatmap(df1, annot=True, fmt="d",  cmap='Blues',  linewidths=.5)

    plt.xlabel('Hospital ID')
    plt.ylabel('Medical Imaging Label')
    plt.tight_layout()
    _save_plot(fig, f"label_distribution_numeric_labels_{fname_type}")


#


def plot_label_distribution(client2class, labelid2labelname, fname_type):
    # client2class = {f'H{k}':v for k, v in client2class.items()}
    client2class = {f'H{k}': v for k, v in client2class.items()}
    plot_label_distribution_heatmap_concrete(
        client2class=client2class, labelid2labelname=labelid2labelname, fname_type=fname_type)

    # _save_plot(fig, "label_distribution")


def _plot_motivation_example_tracefl_output():

    def _plot(plot_dict, ax, title):
        plot_dict = {k.replace('c', 'H'): v for k, v in plot_dict.items()}
        df = pd.DataFrame([plot_dict])
        sns.heatmap(df, annot=True, ax=ax, cmap="YlGnBu",
                    cbar=False, linewidths=.5, linecolor='black')
        ax.set_title(title, fontsize=9)
        ax.set_yticklabels([''])

    fig, axes = _call_before_everyPlot(
        width_inches=3.3374*1.6, height_inches=3.3374/1.318, nrows=4, ncols=1)

    d1 = {'c2': 0.34, 'c4': 0.33, 'c3': 0.25, 'c1': 0.02, 'c7': 0.02,
          'c5': 0.01, 'c6': 0.01, 'c8': 0.01, 'c9': 0.0, 'c0': 0.0}
    d2 = {'c2': 0.34, 'c4': 0.32, 'c3': 0.26, 'c7': 0.02, 'c1': 0.02,
          'c8': 0.01, 'c5': 0.01, 'c6': 0.01, 'c0': 0.0, 'c9': 0.0}
    d3 = {'c6': 0.37, 'c5': 0.32, 'c7': 0.25, 'c3': 0.02, 'c1': 0.01,
          'c8': 0.01, 'c2': 0.01, 'c4': 0.01, 'c9': 0.0, 'c0': 0.0}
    d4 = {'c6': 0.37, 'c5': 0.32, 'c7': 0.25, 'c3': 0.02, 'c1': 0.01,
          'c8': 0.01, 'c2': 0.01, 'c4': 0.01, 'c9': 0.0, 'c0': 0.0}

    _plot(d1, axes[0], 'a) TraceFL output on test image one (Mucus).')
    _plot(d2, axes[1], 'b) TraceFL output on test image two (Mucus).')
    _plot(
        d3, axes[2], 'c) TraceFL output on test image three (Cancer-associated Stroma).')
    _plot(
        d4, axes[3], 'd) TraceFL output on test image four (Cancer-associated Stroma).')

    # set x label using figure
    fig.supxlabel('Hospital ID', fontsize=10)

    plt.tight_layout()
    _save_plot(fig, f"motivation_example_tracefl_output")


def _plot_table_fed_debug_comparison() -> None:
    def _read_df(start_key, mname, dname, faulty_cleint_id, label2flip, data_dist,alpha=None):
        fname = f'results_csvs/prov_{start_key}-{mname}-{dname}-faulty_clients[[{faulty_cleint_id}]]-noise_rate{label2flip}-TClients10-fedavg-(R15-clientsPerR10)-{data_dist}{alpha}-batch32-epochs2-lr0.001.csv'
        if len(fname) > 250:
            hashe_name = get_hashed_name(fname)
            fname = f"results_csvs/{hashe_name}.csv"
        
        return pd.read_csv(fname)

    # def _get_table_dict_pathological(df):
    #     mname = _get_paper_name(df['Model'][0])
    #     dname = _get_paper_name(df['Dataset'][0])
    #     return {'Data Distribution': 'Pathological', 'Model': mname, 'Dataset': dname, 'Avg. FedDebug Time (s)': df['FedDebug avg_fault_localization_time'].mean(), 'Avg. TraceFL Time (s)': df['avg_prov_time_per_input'].mean(), 'FedDebug Accuracy': df['FedDebug Accuracy'].mean(),  'TraceFL Accuracy': df['Accuracy'].mean()*100}
    
    # def _get_table_dict_pathological_text(df):
    #     mname = _get_paper_name(df['Model'][0])
    #     dname = _get_paper_name(df['Dataset'][0])
    #     return {'Data Distribution': 'Pathological', 'Model': mname, 'Dataset': dname, 'Avg. FedDebug Time (s)': 'NA', 'Avg. TraceFL Time (s)': df['avg_prov_time_per_input'].mean(), 'FedDebug Accuracy': "NA",  'TraceFL Accuracy': df['Accuracy'].mean()*100}
    
    def _get_table_dict_dirichlet(df, alpha):
        mname = _get_paper_name(df['Model'][0])
        dname = _get_paper_name(df['Dataset'][0])
        
        if mname.find('GPT') != -1:
            return {'Model': mname, 'Dataset': dname, 'Dirichlet Distribution ($\\alpha$)': alpha,   'Avg. FedDebug Time (s)': "NA", 'Avg. TraceFL Time (s)': df['avg_prov_time_per_input'].mean(), 'FedDebug Accuracy': "NA",  'TraceFL Accuracy': df['Accuracy'].mean()*100}

        else:
            return {'Model': mname, 'Dataset': dname, 'Dirichlet Distribution ($\\alpha$)': alpha,   'Avg. FedDebug Time (s)': df['FedDebug avg_fault_localization_time'].mean(), 'Avg. TraceFL Time (s)': df['avg_prov_time_per_input'].mean(), 'FedDebug Accuracy': df['FedDebug Accuracy'].mean(),  'TraceFL Accuracy': df['Accuracy'].mean()*100}



    l2flip = '{1 0, 2 0, 3 0, 4 0, 5 0, 6 0, 7 0, 8 0, 9 0, 10 0, 11 0, 12 0, 13 0}'
    alphas = [0.3, 0.7, 1]
    dnames = ['organamnist', 'pathmnist', 'cifar10', 'mnist']    

    dirchilet_dfs = []
    summarised_dirchilet_dfs = []
    for dname in dnames:
        for alpha in alphas:
            df_temp = _read_df(start_key='faulty_dirichlet', mname='densenet121', dname=dname, faulty_cleint_id=0, label2flip=l2flip, data_dist='non_iid_dirichlet', alpha=alpha)
            dirchilet_dfs.append(df_temp)
            summarised_dirchilet_dfs.append(_get_table_dict_dirichlet(df_temp, alpha))


    for dname in ['dbpedia_14', 'yahoo_answers_topics']:
        for alpha in alphas:
            df_temp = _read_df(start_key='faulty_dirichlet', mname='openai-communityopenai-gpt', dname=dname, faulty_cleint_id=0, label2flip=l2flip, data_dist='non_iid_dirichlet', alpha=alpha)
            dirchilet_dfs.append(df_temp)
            summarised_dirchilet_dfs.append(_get_table_dict_dirichlet(df_temp, alpha))


        
    all_dicts = summarised_dirchilet_dfs # [_get_table_dict_pathological(df) for df in pathological_dfs] + summarised_dirchilet_dfs + [_get_table_dict_pathological_text(df5), _get_table_dict_pathological_text(df6)]

    df = pd.DataFrame(all_dicts)
    print(df)

    df.to_csv('results_csvs/feddebug_vs_tracefl_time_comparison.csv')

    selected_columns = ['Model',  'Dataset', 'Dirichlet Distribution ($\\alpha$)', 'FedDebug Accuracy', 'TraceFL Accuracy']

    df_latex = df[selected_columns]

    latex_code = df_latex.to_latex(index=False, float_format="%.2f")
    with open("graphs/tables/feddebug_vs_tracefl_comparison.tex", "w") as f:
        f.write(latex_code)

    all_dfs =  dirchilet_dfs  #pathological_dfs + dirchilet_dfs + [df5, df6]

    logging.info(f"-------------- FedDebug vs TraceFL Comparison --------------")

    total_acc = sum([df['Accuracy'].sum()*100 for df in all_dfs])
    total_rounds = sum([len(df) for df in all_dfs])
    total_models_trained = total_rounds * 10
    average_accuracy = total_acc / total_rounds
    logging.info(f"Total Rounds: {total_rounds}")
    logging.info(f"Total Accuracy: {total_acc}")
    logging.info(f"Average Accuracy: {average_accuracy}")
    logging.info(f'Total Models Trained:  {total_models_trained}')


    return {'Total Rounds': total_rounds, 'Total Accuracy': total_acc, 'Average Accuracy': average_accuracy, 'Total Models Trained': total_models_trained}     


def _plot_overhead():


    # Create the DataFrame
    # data = {
    #     "Model-Dataset": [
    #         "Abdominal-CT", "Colon-Pathology",  "DBpedia", "Yahoo-Answers"
    #     ],
    #     "Avg. FedDebug Time (s)": [1.065782728, 1.032155128,  0, 0],
    #     "Avg. TraceFL Time (s)": [3.863399574, 3.904629768,  2.699528438, 2.634212631]
    # }

    # Create the DataFrame
    data = {
        "Model-Dataset": [
            "Abdominal-CT", "Colon-Pathology", "CIFAR10", 
            "MNIST", "DBpedia", "Yahoo-Answers"
        ],
        "Avg. FedDebug Time (s)": [1.065782728, 1.032155128, 1.099321423, 1.136827763, 0, 0],
        "Avg. TraceFL Time (s)": [3.863399574, 3.904629768, 4.61762392, 4.898141942, 2.699528438, 2.634212631]
    }

    df = pd.DataFrame(data)

    # Plotting the bar plot
    # plt.figure(figsize=(10, 6))
    fig, ax =  _call_before_everyPlot(width_inches=3.3374*1.12, height_inches=3.3374/1.4, nrows=1, ncols=1)

    
    sns.barplot(y="value", x="Model-Dataset", hue="variable",
            data=pd.melt(df, id_vars=["Model-Dataset"]), palette="viridis")

    # plt.yticks(rotation=0)
    # plt.title("Comparison of Avg. FedDebug Time and Avg. TraceFL Time across Model-Datasets")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Avg. Time (s)")
    plt.xlabel("Model-Dataset")
    plt.ylim(0, 12)

    # Adjust the legend to remove 'variable'
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=[label.split()[-3] + " " + label.split()[-2] + " (s)" for label in labels], ncol=2, handletextpad=0.01,
               columnspacing=0.1,
               borderpad=0.1,
               labelspacing=0.1,)

    # plt.tight_layout()

    plt.tight_layout()
    plt.xlabel("")
    # plt.show()
    _save_plot(fig, "overhead_comparison")





def do_plotting():
     

    # todo
    logging.info('======================== Section Correct Predictions ========================')
    r1 =  _plot_text_image_audio_classification_results()
    r2 = _table_and_graph_scalability_results()
    # sum r1 and r2 metrics 
    total_rounds_sec1 = r1['Total Rounds'] + r2['Total Rounds']
    total_accuracy_sec1 = r1['Total Accuracy'] + r2['Total Accuracy']
    average_accuracy = total_accuracy_sec1 / total_rounds_sec1
    total_model_trained_sec1 = r1['Total Models Trained'] + r2['Total Models Trained']
    logging.info(f"-------------- Total in Accross {total_rounds_sec1} in 2 sections --------------")
    logging.info(f"Total Rounds: {total_rounds_sec1}")
    logging.info(f"Total Accuracy: {total_accuracy_sec1}")
    logging.info(f"Average Accuracy: {average_accuracy}")
    logging.info(f'Total Models Trained:  {total_model_trained_sec1}')
    



    logging.info('======================== Section Incorrect Predictions and Comparison With FedDebug ========================')
    r3 =  _plot_table_fed_debug_comparison()
    
    # total_rounds_sec2_plus_sect_1 = r3['Total Rounds'] + total_rounds_sec1
    # total_accuracy_sec2_plus_sect_1 = r3['Total Accuracy'] + total_accuracy_sec1
    # average_accuracy_sec2_plus_sect_1 = total_accuracy_sec2_plus_sect_1 / total_rounds_sec2_plus_sect_1
    # total_model_trained_sec2_plus_sect_1 = r3['Total Models Trained'] + total_model_trained_sec1
    # logging.info(f"The total rounds in section 2 and 1 is {total_rounds_sec2_plus_sect_1}")
    # logging.info(f"Total Accuracy Sec2+Sec1: {total_accuracy_sec2_plus_sect_1}")
    # logging.info(f"Average Accuracy Sec2+Sec1: {average_accuracy_sec2_plus_sect_1}")
    # logging.info(f'Total Models Trained Sec2+Sec1:  {total_model_trained_sec2_plus_sect_1}')

   


    logging.info('======================== Section Dirichlet Alpha vs Accuracy ========================')    
    r4 =  _plot_dirchlet_alpha_vs_accuracy()

    logging.info('======================== Section Differential Privacy ========================')
    r5 =  _plot_differential_privacy_results() # convert to table

    # sum all the metrics
    total_rounds = total_rounds_sec1 + r3['Total Rounds'] + r4['Total Rounds'] + r5['Total Rounds']
    total_accuracy = total_accuracy_sec1 + r3['Total Accuracy'] + r4['Total Accuracy'] + r5['Total Accuracy']
    average_accuracy = total_accuracy / total_rounds
    total_model_trained = total_model_trained_sec1 + r3['Total Models Trained'] + r4['Total Models Trained'] + r5['Total Models Trained']
    logging.info(f"******************************************************************************")
    logging.info(f"-------------- Total in Accross all sections --------------")
    logging.info(f"Total Rounds: {total_rounds}")
    logging.info(f"Total Accuracy: {total_accuracy}")
    logging.info(f"Average Accuracy: {average_accuracy}")
    logging.info(f'Total Models Trained:  {total_model_trained}')
    logging.info(f"******************************************************************************")


    # extra plots
    _plot_label_distribution_motivation()
    _plot_motivation_example_tracefl_output()
    _plot_overhead()

















