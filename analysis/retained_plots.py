import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Define a unified strategy mapping
strategy_map = {
    'busy_A': {'label': 'Stubborn Neurons', 'color': 'C0', 'marker': 'o', 'linestyle': '--'},
    'free_A': {'label': 'Plastic Neurons', 'color': 'C1', 'marker': 's', 'linestyle': '-'},
    'busy_H_spec': {'label': 'Specific Neurons (General)', 'color': 'C2', 'marker': '^', 'linestyle': '-.'},
    'spec_B': {'label': 'Specific Neurons', 'color': 'C3', 'marker': 'D', 'linestyle': '-'},
    'busy_B': {'label': 'Candidate Neurons', 'color': 'C4', 'marker': 'P', 'linestyle': '-'},
    'rnd': {'label': 'Random Neurons', 'color': 'C5', 'marker': '*', 'linestyle': ':'},
    'busy_H': {'label': 'Lottery Ticket Neurons', 'color': 'C6', 'marker': 'v', 'linestyle': '--'},
    'free_H': {'label': 'Non-Lottery Neurons', 'color': 'C7', 'marker': 'X', 'linestyle': '-.'}
}

strategy_map_short = {
    'busy_A': {'label': 'Stubborn', 'color': 'C0', 'marker': 'o', 'linestyle': '--'},
    'free_A': {'label': 'Plastic', 'color': 'C1', 'marker': 's', 'linestyle': '-'},
    'busy_H_spec': {'label': 'Specific (General)', 'color': 'C2', 'marker': '^', 'linestyle': '-.'},
    'spec_B': {'label': 'Specific', 'color': 'C3', 'marker': 'D', 'linestyle': '-'},
    'busy_B': {'label': 'Candidate', 'color': 'C4', 'marker': 'P', 'linestyle': '-'},
    'rnd': {'label': 'Random', 'color': 'C5', 'marker': '*', 'linestyle': ':'},
    'busy_H': {'label': 'Lottery', 'color': 'C6', 'marker': 'v', 'linestyle': '--'},
    'free_H': {'label': 'Non-lottery', 'color': 'C7', 'marker': 'X', 'linestyle': '-.'}
}

def plot_pareto_mosaic_10xneurons(filepath, filename, experiment_name, strategies=None, figsize=(18, 8)):
    # Create output directory if it doesn't exist
    output_dir = f"./figures/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Set the matplotlib settings
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 12, # not used
        'axes.titlesize': 22,  # 2000 Neurons etc.
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 24,
        'font.family': 'serif',
        'text.usetex': True,
        'figure.figsize': (10, 6)
    })
  
    # Load data
    #print(f'opening {os.path.join(filepath, filename)}')
    with open(os.path.join(filepath, filename), 'r') as file: 
        data = json.load(file)

    # Extract the number of folds and thresholds
    n_folds = len(data['results'].keys())
    thresholds = sorted([int(t) for t in list(next(iter(data['results'].values()))['cftb'].keys())])

    # Define default strategies if none are provided
    default_strategies = ['busy_A', 'free_A', 'busy_H_spec', 'spec_B', 'busy_B', 'rnd']

    # If no strategies provided, use the default ones
    if strategies is None:
        strategies = default_strategies

    # Initialize results dictionary
    results = {strategy: {'acc_A': [], 'acc_B': [], 'std_acc_A': [], 'std_acc_B': []} for strategy in strategies}

    # Baselines
    fta_accA = []
    ftb_accA = []
    ftb_accB = []

    # Aggregate results across folds
    for fold in data['results'].values():
        fta_accA.append(fold['fta']['acc_A'])
        ftb_accA.append(fold['ftb']['avg_accA'])
        ftb_accB.append(fold['ftb']['avg_accB'])

        for strategy in strategies:
            acc_A = []
            acc_B = []
            for t in thresholds:
                acc_A.append(fold['cftb'][str(t)][strategy]['avg_accA'])
                acc_B.append(fold['cftb'][str(t)][strategy]['avg_accB'])
            results[strategy]['acc_A'].append(acc_A)
            results[strategy]['acc_B'].append(acc_B)

    # Compute means and standard deviations
    for strategy in strategies:
        results[strategy]['mean_acc_A'] = np.mean(results[strategy]['acc_A'], axis=0)
        results[strategy]['mean_acc_B'] = np.mean(results[strategy]['acc_B'], axis=0)
        results[strategy]['std_acc_A'] = np.std(results[strategy]['acc_A'], axis=0)
        results[strategy]['std_acc_B'] = np.std(results[strategy]['acc_B'], axis=0)

    # Baseline means
    fta_accA_mean = np.mean(fta_accA)
    ftb_accA_mean = np.mean(ftb_accA)
    ftb_accB_mean = np.mean(ftb_accB)

    # Plotting
    #sns.set_theme(style="whitegrid", context="paper", font_scale=1.6)

    fig, axes = plt.subplots(2, 5, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    for i, neurons in enumerate(thresholds):
        ax = axes[i]
        
        for strategy in strategies:
            ax.scatter(results[strategy]['mean_acc_A'][i], results[strategy]['mean_acc_B'][i], 
                       label=strategy_map[strategy]['label'], color=strategy_map[strategy]['color'], 
                       s=144, alpha=0.7, marker=strategy_map[strategy]['marker'])

        ax.set_title(f'{neurons} Neurons')
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major')

    fig.text(0.5, 0.02, 'Accuracy on Previous Knowledge', ha='center')
    fig.text(0.02, 0.5, 'Accuracy on New Knowledge', va='center', rotation='vertical')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=6, markerscale=1.7)

    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])

    # Save the plot
    output_path = os.path.join(output_dir,f"{filename[:-5]}_neuron_update_strategies_scatter.pdf")
    plt.savefig(output_path)
    plt.show()

def plot_pareto_mosaic(filepath, filename, experiment_name, strategies=None, figsize=(18, 10)):
    # Create output directory if it doesn't exist
    output_dir = f"./figures/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Set the matplotlib settings
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 12, # not used
        'axes.titlesize': 22,  # 2000 Neurons etc.
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 24,
        'font.family': 'serif',
        'text.usetex': True,
        'figure.figsize': (10, 6)
    })
  
    # Load data
    #print(f'opening {os.path.join(filepath, filename)}')
    with open(os.path.join(filepath, filename), 'r') as file: 
        data = json.load(file)

    # Extract the number of folds and thresholds
    n_folds = len(data['results'].keys())
    thresholds = [int(t) for t in list(next(iter(data['results'].values()))['cftb'].keys())]

    # Define default strategies if none are provided
    default_strategies = ['busy_A', 'free_A', 'busy_H_spec', 'spec_B', 'busy_B', 'rnd']

    # If no strategies provided, use the default ones
    if strategies is None:
        strategies = default_strategies

    # Initialize results dictionary
    results = {strategy: {'acc_A': [], 'acc_B': [], 'std_acc_A': [], 'std_acc_B': []} for strategy in strategies}

    # Baselines
    fta_accA = []
    ftb_accA = []
    ftb_accB = []

    # Aggregate results across folds
    for fold in data['results'].values():
        fta_accA.append(fold['fta']['acc_A'])
        ftb_accA.append(fold['ftb']['avg_accA'])
        ftb_accB.append(fold['ftb']['avg_accB'])

        for strategy in strategies:
            acc_A = []
            acc_B = []
            for t in thresholds:
                acc_A.append(fold['cftb'][str(t)][strategy]['avg_accA'])
                acc_B.append(fold['cftb'][str(t)][strategy]['avg_accB'])
            results[strategy]['acc_A'].append(acc_A)
            results[strategy]['acc_B'].append(acc_B)

    # Compute means and standard deviations
    for strategy in strategies:
        results[strategy]['mean_acc_A'] = np.mean(results[strategy]['acc_A'], axis=0)
        results[strategy]['mean_acc_B'] = np.mean(results[strategy]['acc_B'], axis=0)
        results[strategy]['std_acc_A'] = np.std(results[strategy]['acc_A'], axis=0)
        results[strategy]['std_acc_B'] = np.std(results[strategy]['acc_B'], axis=0)

    # Baseline means
    fta_accA_mean = np.mean(fta_accA)
    ftb_accA_mean = np.mean(ftb_accA)
    ftb_accB_mean = np.mean(ftb_accB)

    # Plotting
    #sns.set_theme(style="whitegrid", context="paper", font_scale=1.6)

    fig, axes = plt.subplots(2, 5, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    for i, neurons in enumerate(thresholds):
        ax = axes[i]
        
        for strategy in strategies:
            ax.scatter(results[strategy]['mean_acc_A'][i], results[strategy]['mean_acc_B'][i], 
                       label=strategy_map[strategy]['label'], color=strategy_map[strategy]['color'], 
                       s=144, alpha=0.7, marker=strategy_map[strategy]['marker'])

        ax.set_title(f'{neurons} Neurons')
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major')

    fig.text(0.5, 0.02, 'Accuracy on Previous Knowledge', ha='center')
    fig.text(0.02, 0.5, 'Accuracy on New Knowledge', va='center', rotation='vertical')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=6, markerscale=1.7)

    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])

    # Save the plot
    output_path = os.path.join(output_dir,f"{filename[:-5]}_neuron_update_strategies_scatter.pdf")
    plt.savefig(output_path)
    plt.show()


import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming strategy_map is defined elsewhere in your code
# If not, you'll need to define it before this function

def plot_old_new_knowledge_all(filepath, filename, experiment_name, strategies=None, loc_old='lower left', bbox_old=(1, 1), loc_new='lower right', bbox_new=(1, 1), y_lim_old_1=0.8, y_lim_old_2 = 1.01):
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'font.family': 'serif',
        'text.usetex': True,
        'figure.figsize': (10, 8)
    })

    # Load data
    with open(os.path.join(filepath, filename), 'r') as file: 
        data = json.load(file)
    
    # Create output directory if it doesn't exist
    output_dir = f"./figures/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)

    # This is likely overriding the plt parameters :(
    #sns.set_theme(style="whitegrid", context="paper", font_scale=1.6)
 
    all_strategies = list(strategy_map_short.keys())

    if strategies is None:
        strategies = all_strategies

    # Initialize fold results only for the selected strategies
    n_folds = len(data['results'].keys())
    
    # Get the keys (thresholds) and sort them
    thresholds = sorted(list(list(data['results'].values())[0]['cftb'].keys()), key=lambda x: float(x))
    n_thresh = len(thresholds)
    
    fold_results = {strategy: {'accA': np.zeros((n_folds, n_thresh)), 'accB': np.zeros((n_folds, n_thresh))} for strategy in strategies}

    fta_accA = 0
    ftb_accA = 0
    ftb_accB = 0

    for i, fold in enumerate(data['results']):
        fta_accA += data['results'][fold]['fta']['acc_A']
        ftb_accA += data['results'][fold]['ftb']['avg_accA']
        ftb_accB += data['results'][fold]['ftb']['avg_accB']

        cftb_results = data['results'][fold]['cftb']

        for j, t in enumerate(thresholds):
            for strategy in strategies:
                fold_results[strategy]['accA'][i, j] = float(cftb_results[t][strategy]['avg_accA'])
                fold_results[strategy]['accB'][i, j] = float(cftb_results[t][strategy]['avg_accB'])

    fta_accA /= n_folds
    ftb_accA /= n_folds
    ftb_accB /= n_folds

    # Calculate means and standard deviations
    means = {strategy: {'accA': np.mean(fold_results[strategy]['accA'], axis=0), 'accB': np.mean(fold_results[strategy]['accB'], axis=0)} for strategy in strategies}
    stds = {strategy: {'accA': np.std(fold_results[strategy]['accA'], axis=0), 'accB': np.std(fold_results[strategy]['accB'], axis=0)} for strategy in strategies}

    # Use the sorted thresholds as x_labels
    x_labels = thresholds

    # Plot - Accuracy of Previous Knowledge
    plt.figure(figsize=(10, 6))
    plt.axhline(y=fta_accA, color='firebrick', linestyle='solid', linewidth=1.5, alpha=0.80, label='Initial')
    plt.axhline(y=ftb_accA, color='dimgrey', linestyle='solid', linewidth=1.5, alpha=0.80, label='Full FT')
    
    for strategy in strategies:
        plt.errorbar(x_labels, means[strategy]['accA'], yerr=stds[strategy]['accA'], 
                     linestyle=strategy_map_short[strategy]['linestyle'], linewidth=1.5, 
                     color=strategy_map_short[strategy]['color'], marker=strategy_map_short[strategy]['marker'], 
                     markersize=10, capsize=5, elinewidth=1.5, capthick=1.5, label=strategy_map_short[strategy]['label'])

    plt.xlabel('Number of updated neurons')
    plt.ylabel('Accuracy')
    plt.legend(loc=loc_old, bbox_to_anchor=bbox_old)
    plt.grid(True)
    plt.ylim(y_lim_old_1,y_lim_old_2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename[:-5]}_neuron_update_strategies_old_knowledge.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    # Plot - Accuracy of New Knowledge
    plt.figure(figsize=(10, 6))
    plt.axhline(y=ftb_accB, color='dimgrey', linestyle='solid', linewidth=1.5, alpha=0.80, label='Full FT')
    
    for strategy in strategies:
        plt.errorbar(x_labels, means[strategy]['accB'], yerr=stds[strategy]['accB'], 
                     linestyle=strategy_map_short[strategy]['linestyle'], linewidth=1.5, 
                     color=strategy_map_short[strategy]['color'], marker=strategy_map_short[strategy]['marker'], 
                     markersize=10, capsize=5, elinewidth=1.5, capthick=1.5, label=strategy_map_short[strategy]['label'])

    plt.xlabel('Number of updated neurons')
    plt.ylabel('Accuracy')
    plt.legend(loc=loc_new, bbox_to_anchor=bbox_new)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename[:-5]}_neuron_update_strategies_new_knowledge.pdf", format='pdf', bbox_inches='tight')
    plt.show()

def plot_old_new_knowledge_all_loRA(filepath, filename, experiment_name, strategies=None, loc_old='lower left', bbox_old=(1, 1), loc_new='lower right', bbox_new=(1, 1), y_lim_old_1=0.8, y_lim_old_2 = 1.01):
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'font.family': 'serif',
        'text.usetex': True,
        'figure.figsize': (10, 8)
    })

    # Load data
    with open(os.path.join(filepath, filename), 'r') as file: 
        data = json.load(file)
    
    # Create output directory if it doesn't exist
    output_dir = f"./figures/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)

    # This is likely overriding the plt parameters :(
    #sns.set_theme(style="whitegrid", context="paper", font_scale=1.6)
 
    all_strategies = list(strategy_map_short.keys())

    if strategies is None:
        strategies = all_strategies

    # Initialize fold results only for the selected strategies
    n_folds = len(data['results'].keys())
    
    # Get the keys (thresholds) and sort them
    thresholds = sorted(list(list(data['results'].values())[0]['cftb'].keys()), key=lambda x: float(x))
    n_thresh = len(thresholds)
    
    fold_results = {strategy: {'accA': np.zeros((n_folds, n_thresh)), 'accB': np.zeros((n_folds, n_thresh))} for strategy in strategies}

    fta_accA = 0
    ftb_accA = 0
    ftb_accB = 0
    LoRAftb_accA = 0
    LoRAftb_accB = 0

    for i, fold in enumerate(data['results']):
        fta_accA += data['results'][fold]['fta']['acc_A']
        ftb_accA += data['results'][fold]['ftb']['avg_accA']
        ftb_accB += data['results'][fold]['ftb']['avg_accB'] 

        LoRAftb_accA += data['results'][fold]['LoRA-ftb']['avg_accA']
        LoRAftb_accB += data['results'][fold]['LoRA-ftb']['avg_accB'] 

        cftb_results = data['results'][fold]['cftb']

        for j, t in enumerate(thresholds):
            for strategy in strategies:
                fold_results[strategy]['accA'][i, j] = float(cftb_results[t][strategy]['avg_accA'])
                fold_results[strategy]['accB'][i, j] = float(cftb_results[t][strategy]['avg_accB'])

    fta_accA /= n_folds
    ftb_accA /= n_folds
    ftb_accB /= n_folds
    LoRAftb_accA /= n_folds
    LoRAftb_accB /= n_folds

    # Calculate means and standard deviations
    means = {strategy: {'accA': np.mean(fold_results[strategy]['accA'], axis=0), 'accB': np.mean(fold_results[strategy]['accB'], axis=0)} for strategy in strategies}
    stds = {strategy: {'accA': np.std(fold_results[strategy]['accA'], axis=0), 'accB': np.std(fold_results[strategy]['accB'], axis=0)} for strategy in strategies}

    # Use the sorted thresholds as x_labels
    x_labels = thresholds

    # Plot - Accuracy of Previous Knowledge
    plt.figure(figsize=(10, 6))
    plt.axhline(y=fta_accA, color='firebrick', linestyle='solid', linewidth=1.5, alpha=0.80, label='Initial')
    plt.axhline(y=ftb_accA, color='dimgrey', linestyle='solid', linewidth=1.5, alpha=0.80, label='Full FT')
    plt.axhline(y=LoRAftb_accA, color='crimson', linestyle=(0, (5, 2, 1, 2)), linewidth=2.5, alpha=0.80, label='LoRA')
    
    for strategy in strategies:
        plt.errorbar(x_labels, means[strategy]['accA'], yerr=stds[strategy]['accA'], 
                     linestyle=strategy_map_short[strategy]['linestyle'], linewidth=1.5, 
                     color=strategy_map_short[strategy]['color'], marker=strategy_map_short[strategy]['marker'], 
                     markersize=10, capsize=5, elinewidth=1.5, capthick=1.5, label=strategy_map_short[strategy]['label'])

    plt.xlabel('Number of updated neurons')
    plt.ylabel('Accuracy')
    #plt.legend(loc=loc_old, bbox_to_anchor=bbox_old)
    plt.grid(True)
    plt.ylim(y_lim_old_1,y_lim_old_2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename[:-5]}_neuron_update_strategies_old_knowledge_LoRa.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    # Plot - Accuracy of New Knowledge
    plt.figure(figsize=(10, 6))
    plt.axhline(y=ftb_accB, color='dimgrey', linestyle='solid', linewidth=1.5, alpha=0.80, label='Full FT')
    plt.axhline(y=LoRAftb_accB, color='crimson', linestyle=(0, (5, 2, 1, 2)), linewidth=2.5, alpha=0.80, label='LoRA')

    
    for strategy in strategies:
        plt.errorbar(x_labels, means[strategy]['accB'], yerr=stds[strategy]['accB'], 
                     linestyle=strategy_map_short[strategy]['linestyle'], linewidth=1.5, 
                     color=strategy_map_short[strategy]['color'], marker=strategy_map_short[strategy]['marker'], 
                     markersize=10, capsize=5, elinewidth=1.5, capthick=1.5, label=strategy_map_short[strategy]['label'])

    plt.xlabel('Number of updated neurons')
    plt.ylabel('Accuracy')
    plt.legend(loc=loc_new, bbox_to_anchor=bbox_new)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename[:-5]}_neuron_update_strategies_new_knowledge_LoRa.pdf", format='pdf', bbox_inches='tight')
    plt.show()

# Function to plot old and new knowledge accuracies
def plot_new_knowledge_all(filepath, filename, experiment_name, strategies=None, loc_new='lower right', bbox_new=(1, 1)):
    # Set up the plot style
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 26,
        'axes.titlesize': 24,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 22,
        'font.family': 'serif',
        'text.usetex': True,
        'figure.figsize': (10, 8)
    })

    # Load data
    with open(os.path.join(filepath, filename), 'r') as file: 
        data = json.load(file)
    
    # Create output directory if it doesn't exist
    output_dir = f"./figures/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
 
 
    all_strategies = list(strategy_map_short.keys())

    if strategies is None:
        strategies = all_strategies

    n_folds =  len(data['results'].keys())
    n_thresh = len(list(data['results'].values())[0]['cfta'].keys())


    # Initialize accumulators for standard deviation calculations
    cftb_stub_accB_list = np.zeros((n_folds, n_thresh))
    cftb_free_accB_list = np.zeros((n_folds, n_thresh))
    cftb_rnd_accB_list = np.zeros((n_folds, n_thresh))

    ftb_accB = 0

    for i, fold in enumerate(data['results']):
        ftb_accB += data['results'][fold]['fta']['acc_A']

        cftb_results = data['results'][fold]['cfta']

        cftb_stub_accB_list[i, :] = np.array([cftb_results[t]['busy_H']['avg_accA'] for t in cftb_results]).astype(float)
        cftb_free_accB_list[i, :] = np.array([cftb_results[t]['free_H']['avg_accA'] for t in cftb_results]).astype(float)
        cftb_rnd_accB_list[i, :] = np.array([cftb_results[t]['rnd']['avg_accA'] for t in cftb_results]).astype(float)

    ftb_accB /= n_folds
    cftb_stub_accB = np.mean(cftb_stub_accB_list, axis=0)
    cftb_free_accB = np.mean(cftb_free_accB_list, axis=0)
    cftb_rnd_accB = np.mean(cftb_rnd_accB_list, axis=0)

    # Calculate standard deviations
    cftb_stub_accB_std = np.std(cftb_stub_accB_list, axis=0)
    cftb_free_accB_std = np.std(cftb_free_accB_list, axis=0)
    cftb_rnd_accB_std = np.std(cftb_rnd_accB_list, axis=0)

    # Creating custom x labels
    x_labels = list(cftb_results.keys())


    # Plot - Accuracy of New Knowledge
    plt.figure(figsize=(10, 6))
    plt.axhline(y=ftb_accB, color='dimgrey', linestyle='solid', linewidth=1.5, alpha=0.80, label='Full FT')
    
    plt.errorbar(x_labels, cftb_free_accB, yerr=cftb_free_accB_std, linestyle=strategy_map_short['free_H']['linestyle'], linewidth=1.5, color=strategy_map_short['free_H']['color'], marker=strategy_map_short['free_H']['marker'], 
                 markersize=10, capsize=5, elinewidth=1.5, capthick=1.5, label=strategy_map_short['free_H']['label'])
    plt.errorbar(x_labels, cftb_stub_accB, yerr=cftb_stub_accB_std, linestyle=strategy_map_short['busy_H']['linestyle'], linewidth=1.5, color=strategy_map_short['busy_H']['color'], marker=strategy_map_short['busy_H']['marker'], 
                 markersize=10, capsize=5, elinewidth=1.5, capthick=1.5, label=strategy_map_short['busy_H']['label'])
    plt.errorbar(x_labels, cftb_rnd_accB, yerr=cftb_rnd_accB_std, linestyle=strategy_map_short['rnd']['linestyle'], linewidth=1.5, color=strategy_map_short['rnd']['color'], marker=strategy_map_short['rnd']['marker'], 
                 markersize=10, capsize=5, elinewidth=1.5, capthick=1.5, label=strategy_map_short['rnd']['label'])


    plt.xlabel('Number of updated neurons')
    plt.ylabel('Accuracy')
    plt.legend(loc=loc_new, bbox_to_anchor=bbox_new)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename[:-5]}_neuron_update_strategies_new_knowledge.pdf", format='pdf', bbox_inches='tight')
    plt.show()

def plot_stubborn_neurons_histogram(data, threshold, output_dir, experiment_name):
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'font.size': 22,
        'axes.labelsize': 22,
        'axes.titlesize': 24,
        'xtick.labelsize': 16,
        'ytick.labelsize': 22,
        'legend.fontsize': 22,
        'font.family': 'serif',
        'text.usetex': True,
        'figure.figsize': (18, 6)
    })

    x_labels = [f"{k.split('.')[-2]}.{k.split('.')[-1]}.{k.split('.')[-3]}" for k in data.keys()]
    lengths = [(len(v[0])) / threshold for v in data.values()]

    # Color-blind friendly palette
    colors = ['#0077BB', '#33BBEE', '#009988', '#EE7733']
    color_labels = ['attn\_cross', 'attn\_proj', 'mlp\_fc', 'mlp\_proj']

    fig, ax = plt.subplots(1, 1, figsize=(18, 6))

    positions = []
    gap = 2
    for i in range(len(lengths)):
        group = i // 4
        pos = i + group * gap
        positions.append(pos)

    bars = ax.bar(positions, lengths, color=[colors[i % len(colors)] for i in range(len(lengths))])

    ax.set_ylabel('Percentage')
    ax.set_xlabel('Transformer blocks')

    group_centers = [(1.5 + i * 4 + i * gap) for i in range(48)]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([str(i + 1) for i in range(48)])

    patches = [mpatches.Patch(color=colors[i], label=color_labels[i]) for i in range(len(colors))]
    ax.legend(handles=patches, loc='upper right')

    ax.grid(True, alpha=0.5)

    plt.tight_layout()

    output_file = f"{output_dir}/{experiment_name}_stubborn_neurons_histogram_{threshold}.pdf"
    print(f"Saving figure to {output_file}")
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()



def plot_editing_old_new_general_bug(experiment_name, filename, filepath, strategies, loc_old, bbox_old,loc_new, bbox_new, loc_gen, bbox_gen, y_lim_old_1, y_lim_old_2):
    # Set up the plot style
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'font.size': 30,
        'axes.labelsize': 34,
        'axes.titlesize': 30,
        'xtick.labelsize': 30,
        'ytick.labelsize': 32,
        'legend.fontsize': 28,
        'font.family': 'serif',
        'text.usetex': True,
        'figure.figsize': (10, 8)
    })
    # Create output directory if it doesn't exist
    output_dir = f"./figures/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    # Load data
    with open(os.path.join(filepath, filename), 'r') as file: 
        data = json.load(file)
    # with open(f'../results/experiment_3_1/experiment_3_1_2000_{N_SAMPLES}.json', 'r') as file: 
    #     data = json.load(file)

    results = data['results']

    n_folds =  len(data['results'].keys())
    n_thresh = len(list(data['results'].values())[0]['cft-notb'].keys())

    fta_accA = 0

    ftb_accA = 0
    ftb_accB = 0
    Loraftb_accA = 0
    Loraftb_accB = 0
    ftb_acc_gen = 0
    

    # Arrays to store individual fold results for calculating std dev
    fold_results_cftb_busyA_accA = np.zeros((n_folds, n_thresh))
    fold_results_cftb_busyA_accB = np.zeros((n_folds, n_thresh))
    fold_results_cftb_busyA_gen = np.zeros((n_folds, n_thresh))

    fold_results_cftb_freeA_accA = np.zeros((n_folds, n_thresh))
    fold_results_cftb_freeA_accB = np.zeros((n_folds, n_thresh))
    fold_results_cftb_freeA_gen = np.zeros((n_folds, n_thresh))

    fold_results_cftb_busyB_accA = np.zeros((n_folds, n_thresh))
    fold_results_cftb_busyB_accB = np.zeros((n_folds, n_thresh))
    fold_results_cftb_busyB_gen = np.zeros((n_folds, n_thresh))

    fold_results_cftb_freeB_accA = np.zeros((n_folds, n_thresh))
    fold_results_cftb_freeB_accB = np.zeros((n_folds, n_thresh))
    fold_results_cftb_freeB_gen = np.zeros((n_folds, n_thresh))

    fold_results_cftb_specB_accA = np.zeros((n_folds, n_thresh))
    fold_results_cftb_specB_accB = np.zeros((n_folds, n_thresh))
    fold_results_cftb_specB_gen = np.zeros((n_folds, n_thresh))

    fold_results_cftb_rnd_accA = np.zeros((n_folds, n_thresh))
    fold_results_cftb_rnd_accB = np.zeros((n_folds, n_thresh))
    fold_results_cftb_rnd_gen = np.zeros((n_folds, n_thresh))

    for i, fold in enumerate(data['results']):
        # Was unfortunately missing from the initial analysis :( 
        new_old_knowledge = data['results'][fold]['ftb']['acc_A']
        # ftb_accA += data['results'][fold]['ft-notb']['avg_accA'] => change to:
        ftb_accA += (data['results'][fold]['ft-notb']['avg_accA']/float(new_old_knowledge))
        ftb_accB += data['results'][fold]['ft-notb']['avg_accB']

        #Loraftb_accA += data['results'][fold]['LoRA-ft-notb']['avg_accA']
        Loraftb_accA += (data['results'][fold]['LoRA-ft-notb']['avg_accA']/float(new_old_knowledge))
        Loraftb_accB += data['results'][fold]['LoRA-ft-notb']['avg_accB']

        cftb_results = data['results'][fold]['cft-notb']

        fold_results_cftb_busyA_accA[i, :] = np.array([cftb_results[t]['busy_A']['avg_accA']/float(new_old_knowledge) for t in cftb_results]).astype(float)
        fold_results_cftb_busyA_accB[i, :] = np.array([cftb_results[t]['busy_A']['avg_accB'] for t in cftb_results]).astype(float)
        fold_results_cftb_busyA_gen[i, :] = np.array([cftb_results[t]['busy_A']['avg_acc_gen'] for t in cftb_results]).astype(float)

        fold_results_cftb_busyB_accA[i, :] = np.array([cftb_results[t]['busy_B']['avg_accA']/float(new_old_knowledge) for t in cftb_results]).astype(float)
        fold_results_cftb_busyB_accB[i, :] = np.array([cftb_results[t]['busy_B']['avg_accB'] for t in cftb_results]).astype(float)
        fold_results_cftb_busyB_gen[i, :] = np.array([cftb_results[t]['busy_B']['avg_acc_gen'] for t in cftb_results]).astype(float)

        fold_results_cftb_freeA_accA[i, :] = np.array([cftb_results[t]['free_A']['avg_accA']/float(new_old_knowledge) for t in cftb_results]).astype(float)
        fold_results_cftb_freeA_accB[i, :] = np.array([cftb_results[t]['free_A']['avg_accB'] for t in cftb_results]).astype(float)
        fold_results_cftb_freeA_gen[i, :] = np.array([cftb_results[t]['free_A']['avg_acc_gen'] for t in cftb_results]).astype(float)

        fold_results_cftb_freeB_accA[i, :] = np.array([cftb_results[t]['free_B']['avg_accA']/float(new_old_knowledge) for t in cftb_results]).astype(float)
        fold_results_cftb_freeB_accB[i, :] = np.array([cftb_results[t]['free_B']['avg_accB'] for t in cftb_results]).astype(float)
        fold_results_cftb_freeB_gen[i, :] = np.array([cftb_results[t]['free_B']['avg_acc_gen'] for t in cftb_results]).astype(float)

        fold_results_cftb_specB_accA[i, :] = np.array([cftb_results[t]['spec_B']['avg_accA']/float(new_old_knowledge) for t in cftb_results]).astype(float)
        fold_results_cftb_specB_accB[i, :] = np.array([cftb_results[t]['spec_B']['avg_accB'] for t in cftb_results]).astype(float)
        fold_results_cftb_specB_gen[i, :] = np.array([cftb_results[t]['spec_B']['avg_acc_gen'] for t in cftb_results]).astype(float)

        fold_results_cftb_rnd_accA[i, :] = np.array([cftb_results[t]['rnd']['avg_accA']/float(new_old_knowledge) for t in cftb_results]).astype(float)
        fold_results_cftb_rnd_accB[i, :] = np.array([cftb_results[t]['rnd']['avg_accB'] for t in cftb_results]).astype(float)
        fold_results_cftb_rnd_gen[i, :] = np.array([cftb_results[t]['rnd']['avg_acc_gen'] for t in cftb_results]).astype(float)

    fta_accA /= n_folds
    ftb_accA /= n_folds
    ftb_accB /= n_folds
    Loraftb_accA /= n_folds
    Loraftb_accB /= n_folds    
    

    cftb_freeA_accA = np.mean(fold_results_cftb_freeA_accA, axis=0)
    cftb_freeA_accB = np.mean(fold_results_cftb_freeA_accB, axis=0)
    cftb_freeA_gen = np.mean(fold_results_cftb_freeA_gen, axis=0)

    cftb_freeB_accA = np.mean(fold_results_cftb_freeB_accA, axis=0)
    cftb_freeB_accB = np.mean(fold_results_cftb_freeB_accB, axis=0)
    cftb_freeB_gen = np.mean(fold_results_cftb_freeB_gen, axis=0)

    cftb_busyA_accA = np.mean(fold_results_cftb_busyA_accA, axis=0)
    cftb_busyA_accB = np.mean(fold_results_cftb_busyA_accB, axis=0)
    cftb_busyA_gen = np.mean(fold_results_cftb_busyA_gen, axis=0)

    cftb_busyB_accA = np.mean(fold_results_cftb_busyB_accA, axis=0)
    cftb_busyB_accB = np.mean(fold_results_cftb_busyB_accB, axis=0)
    cftb_busyB_gen = np.mean(fold_results_cftb_busyB_gen, axis=0)

    cftb_specB_accA = np.mean(fold_results_cftb_specB_accA, axis=0)
    cftb_specB_accB = np.mean(fold_results_cftb_specB_accB, axis=0)
    cftb_specB_gen = np.mean(fold_results_cftb_specB_gen, axis=0)

    cftb_rnd_accA = np.mean(fold_results_cftb_rnd_accA, axis=0)
    cftb_rnd_accB = np.mean(fold_results_cftb_rnd_accB, axis=0)
    cftb_rnd_gen = np.mean(fold_results_cftb_rnd_gen, axis=0)

    # Calculate standard deviation
    std_cftb_freeA_accA = np.std(fold_results_cftb_freeA_accA, axis=0)
    std_cftb_freeA_accB = np.std(fold_results_cftb_freeA_accB, axis=0)
    std_cftb_freeA_gen = np.std(fold_results_cftb_freeA_gen, axis=0)

    std_cftb_freeB_accA = np.std(fold_results_cftb_freeB_accA, axis=0)
    std_cftb_freeB_accB = np.std(fold_results_cftb_freeB_accB, axis=0)
    std_cftb_freeB_gen = np.std(fold_results_cftb_freeB_gen, axis=0)

    std_cftb_busyA_accA = np.std(fold_results_cftb_busyA_accA, axis=0)
    std_cftb_busyA_accB = np.std(fold_results_cftb_busyA_accB, axis=0)
    std_cftb_busyA_gen = np.std(fold_results_cftb_busyA_gen, axis=0)

    std_cftb_busyB_accA = np.std(fold_results_cftb_busyB_accA, axis=0)
    std_cftb_busyB_accB = np.std(fold_results_cftb_busyB_accB, axis=0)
    std_cftb_busyB_gen = np.std(fold_results_cftb_busyB_gen, axis=0)

    std_cftb_specB_accA = np.std(fold_results_cftb_specB_accA, axis=0)
    std_cftb_specB_accB = np.std(fold_results_cftb_specB_accB, axis=0)
    std_cftb_specB_gen = np.std(fold_results_cftb_specB_gen, axis=0)

    std_cftb_rnd_accA = np.std(fold_results_cftb_rnd_accA, axis=0)
    std_cftb_rnd_accB = np.std(fold_results_cftb_rnd_accB, axis=0)
    std_cftb_rnd_gen = np.std(fold_results_cftb_rnd_gen, axis=0)

    #fig, axs = plt.subplots(1, 1, figsize=(12, 6))
    # Plot - Accuracy of Previous Knowledge
    plt.figure(figsize=(12, 6))
    plt.axhline(y=ftb_accA, color='dimgrey', linestyle='solid', linewidth=1.5, alpha=0.80, label='Full FT')
    plt.axhline(y=Loraftb_accA, color='crimson', linestyle=(0, (5, 2, 1, 2)), linewidth=2.5, alpha=0.80, label='LoRA')


    # Creating custom x labels
    x_labels = list(cftb_results.keys())

    # First plotl 
    plt.errorbar(x_labels, cftb_busyB_accA, yerr=std_cftb_busyB_accA, **strategy_map_short['busy_B'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_freeA_accA, yerr=std_cftb_freeA_accA, **strategy_map_short['free_A'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_specB_accA, yerr=std_cftb_specB_accA, **strategy_map_short['spec_B'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_rnd_accA, yerr=std_cftb_rnd_accA, **strategy_map_short['rnd'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.xlabel('Number of updated neurons')
    plt.ylabel('Accuracy')
    #axs.set_title('Sequential learning experiment - Accuracy of Previous Knowledge')
    #plt.legend(loc=loc_old, bbox_to_anchor=bbox_old)
    plt.grid(True)
    plt.ylim(y_lim_old_1, y_lim_old_2)
    plt.tight_layout()
    plt.xticks(ticks=x_labels, labels=[f"{int(int(x)/1000)}k" for x in x_labels])
    plt.savefig(f"{output_dir}/{filename[:-5]}_neuron_update_strategies_old_knowledge.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    # Plot - Accuracy of New Knowledge
    plt.figure(figsize=(12, 6))
    plt.axhline(y=ftb_accB, color='dimgrey', linestyle='solid', linewidth=1.5, alpha=0.80, label='Full FT')
    plt.axhline(y=Loraftb_accB, color='crimson', linestyle=(0, (5, 2, 1, 2)), linewidth=2.5, alpha=0.80, label='LoRA')
    plt.errorbar(x_labels, cftb_busyB_accB, yerr=std_cftb_busyB_accB, **strategy_map_short['busy_B'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_freeA_accB, yerr=std_cftb_freeA_accB, **strategy_map_short['free_A'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_specB_accB, yerr=std_cftb_specB_accB, **strategy_map_short['spec_B'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_rnd_accB, yerr=std_cftb_rnd_accB, **strategy_map_short['rnd'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.xlabel('Number of updated neurons')
    plt.ylabel('Accuracy')
    plt.legend(loc=loc_new, bbox_to_anchor=bbox_new)
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(ticks=x_labels, labels=[f"{int(int(x)/1000)}k" for x in x_labels])
    plt.savefig(f"{output_dir}/{filename[:-5]}_neuron_update_strategies_new_knowledge.pdf", format='pdf', bbox_inches='tight')
    plt.show()


    #plt.figure(figsize=(12, 6))
    plt.figure(figsize=(12, 6))

    plt.axhline(y=ftb_accB, color='dimgrey', linestyle='solid', linewidth=1.5, alpha=0.80, label='Full FT')

    plt.errorbar(x_labels, cftb_busyB_gen, yerr=std_cftb_busyB_gen, **strategy_map_short['busy_B'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_freeA_gen, yerr=std_cftb_freeA_gen, **strategy_map_short['free_A'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_specB_gen, yerr=std_cftb_specB_gen, **strategy_map_short['spec_B'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_rnd_gen, yerr=std_cftb_rnd_gen, **strategy_map_short['rnd'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)

    plt.xlabel('Number of updated neurons')
    plt.ylabel('Accuracy')
    #plt.legend(loc=loc_gen, bbox_to_anchor=bbox_gen)
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(ticks=x_labels, labels=[f"{int(int(x)/1000)}k" for x in x_labels])
    plt.savefig(f"{output_dir}/{filename[:-5]}_neuron_update_strategies_general_knowledge.pdf", format='pdf', bbox_inches='tight')
    plt.show()

def plot_editing_old_new_general(experiment_name, filename, filepath, strategies, loc_old, bbox_old,loc_new, bbox_new, loc_gen, bbox_gen, y_lim_old_1, y_lim_old_2):
    # Set up the plot style
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'font.size': 30,
        'axes.labelsize': 34,
        'axes.titlesize': 30,
        'xtick.labelsize': 30,
        'ytick.labelsize': 32,
        'legend.fontsize': 28,
        'font.family': 'serif',
        'text.usetex': True,
        'figure.figsize': (10, 8)
    })
    # Create output directory if it doesn't exist
    output_dir = f"./figures/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    # Load data
    with open(os.path.join(filepath, filename), 'r') as file: 
        data = json.load(file)
    # with open(f'../results/experiment_3_1/experiment_3_1_2000_{N_SAMPLES}.json', 'r') as file: 
    #     data = json.load(file)

    results = data['results']

    n_folds =  len(data['results'].keys())
    n_thresh = len(list(data['results'].values())[0]['cft-notb'].keys())

    fta_accA = 0

    ftb_accA = 0
    ftb_accB = 0
    Loraftb_accA = 0
    Loraftb_accB = 0
    ftb_acc_gen = 0
    

    # Arrays to store individual fold results for calculating std dev
    fold_results_cftb_busyA_accA = np.zeros((n_folds, n_thresh))
    fold_results_cftb_busyA_accB = np.zeros((n_folds, n_thresh))
    fold_results_cftb_busyA_gen = np.zeros((n_folds, n_thresh))

    fold_results_cftb_freeA_accA = np.zeros((n_folds, n_thresh))
    fold_results_cftb_freeA_accB = np.zeros((n_folds, n_thresh))
    fold_results_cftb_freeA_gen = np.zeros((n_folds, n_thresh))

    fold_results_cftb_busyB_accA = np.zeros((n_folds, n_thresh))
    fold_results_cftb_busyB_accB = np.zeros((n_folds, n_thresh))
    fold_results_cftb_busyB_gen = np.zeros((n_folds, n_thresh))

    fold_results_cftb_freeB_accA = np.zeros((n_folds, n_thresh))
    fold_results_cftb_freeB_accB = np.zeros((n_folds, n_thresh))
    fold_results_cftb_freeB_gen = np.zeros((n_folds, n_thresh))

    fold_results_cftb_specB_accA = np.zeros((n_folds, n_thresh))
    fold_results_cftb_specB_accB = np.zeros((n_folds, n_thresh))
    fold_results_cftb_specB_gen = np.zeros((n_folds, n_thresh))

    fold_results_cftb_rnd_accA = np.zeros((n_folds, n_thresh))
    fold_results_cftb_rnd_accB = np.zeros((n_folds, n_thresh))
    fold_results_cftb_rnd_gen = np.zeros((n_folds, n_thresh))

    for i, fold in enumerate(data['results']):
        ftb_accA += data['results'][fold]['ft-notb']['avg_accA']
        ftb_accB += data['results'][fold]['ft-notb']['avg_accB']

        #Loraftb_accA += data['results'][fold]['LoRA-ft-notb']['avg_accA']
        Loraftb_accA += (data['results'][fold]['LoRA-ft-notb']['avg_accA'])
        Loraftb_accB += data['results'][fold]['LoRA-ft-notb']['avg_accB']

        cftb_results = data['results'][fold]['cft-notb']

        fold_results_cftb_busyA_accA[i, :] = np.array([cftb_results[t]['busy_A']['avg_accA'] for t in cftb_results]).astype(float)
        fold_results_cftb_busyA_accB[i, :] = np.array([cftb_results[t]['busy_A']['avg_accB'] for t in cftb_results]).astype(float)
        fold_results_cftb_busyA_gen[i, :] = np.array([cftb_results[t]['busy_A']['avg_acc_gen'] for t in cftb_results]).astype(float)

        fold_results_cftb_busyB_accA[i, :] = np.array([cftb_results[t]['busy_B']['avg_accA'] for t in cftb_results]).astype(float)
        fold_results_cftb_busyB_accB[i, :] = np.array([cftb_results[t]['busy_B']['avg_accB'] for t in cftb_results]).astype(float)
        fold_results_cftb_busyB_gen[i, :] = np.array([cftb_results[t]['busy_B']['avg_acc_gen'] for t in cftb_results]).astype(float)

        fold_results_cftb_freeA_accA[i, :] = np.array([cftb_results[t]['free_A']['avg_accA'] for t in cftb_results]).astype(float)
        fold_results_cftb_freeA_accB[i, :] = np.array([cftb_results[t]['free_A']['avg_accB'] for t in cftb_results]).astype(float)
        fold_results_cftb_freeA_gen[i, :] = np.array([cftb_results[t]['free_A']['avg_acc_gen'] for t in cftb_results]).astype(float)

        fold_results_cftb_freeB_accA[i, :] = np.array([cftb_results[t]['free_B']['avg_accA'] for t in cftb_results]).astype(float)
        fold_results_cftb_freeB_accB[i, :] = np.array([cftb_results[t]['free_B']['avg_accB'] for t in cftb_results]).astype(float)
        fold_results_cftb_freeB_gen[i, :] = np.array([cftb_results[t]['free_B']['avg_acc_gen'] for t in cftb_results]).astype(float)

        fold_results_cftb_specB_accA[i, :] = np.array([cftb_results[t]['spec_B']['avg_accA'] for t in cftb_results]).astype(float)
        fold_results_cftb_specB_accB[i, :] = np.array([cftb_results[t]['spec_B']['avg_accB'] for t in cftb_results]).astype(float)
        fold_results_cftb_specB_gen[i, :] = np.array([cftb_results[t]['spec_B']['avg_acc_gen'] for t in cftb_results]).astype(float)

        fold_results_cftb_rnd_accA[i, :] = np.array([cftb_results[t]['rnd']['avg_accA'] for t in cftb_results]).astype(float)
        fold_results_cftb_rnd_accB[i, :] = np.array([cftb_results[t]['rnd']['avg_accB'] for t in cftb_results]).astype(float)
        fold_results_cftb_rnd_gen[i, :] = np.array([cftb_results[t]['rnd']['avg_acc_gen'] for t in cftb_results]).astype(float)

    fta_accA /= n_folds
    ftb_accA /= n_folds
    ftb_accB /= n_folds
    Loraftb_accA /= n_folds
    Loraftb_accB /= n_folds    
    

    cftb_freeA_accA = np.mean(fold_results_cftb_freeA_accA, axis=0)
    cftb_freeA_accB = np.mean(fold_results_cftb_freeA_accB, axis=0)
    cftb_freeA_gen = np.mean(fold_results_cftb_freeA_gen, axis=0)

    cftb_freeB_accA = np.mean(fold_results_cftb_freeB_accA, axis=0)
    cftb_freeB_accB = np.mean(fold_results_cftb_freeB_accB, axis=0)
    cftb_freeB_gen = np.mean(fold_results_cftb_freeB_gen, axis=0)

    cftb_busyA_accA = np.mean(fold_results_cftb_busyA_accA, axis=0)
    cftb_busyA_accB = np.mean(fold_results_cftb_busyA_accB, axis=0)
    cftb_busyA_gen = np.mean(fold_results_cftb_busyA_gen, axis=0)

    cftb_busyB_accA = np.mean(fold_results_cftb_busyB_accA, axis=0)
    cftb_busyB_accB = np.mean(fold_results_cftb_busyB_accB, axis=0)
    cftb_busyB_gen = np.mean(fold_results_cftb_busyB_gen, axis=0)

    cftb_specB_accA = np.mean(fold_results_cftb_specB_accA, axis=0)
    cftb_specB_accB = np.mean(fold_results_cftb_specB_accB, axis=0)
    cftb_specB_gen = np.mean(fold_results_cftb_specB_gen, axis=0)

    cftb_rnd_accA = np.mean(fold_results_cftb_rnd_accA, axis=0)
    cftb_rnd_accB = np.mean(fold_results_cftb_rnd_accB, axis=0)
    cftb_rnd_gen = np.mean(fold_results_cftb_rnd_gen, axis=0)

    # Calculate standard deviation
    std_cftb_freeA_accA = np.std(fold_results_cftb_freeA_accA, axis=0)
    std_cftb_freeA_accB = np.std(fold_results_cftb_freeA_accB, axis=0)
    std_cftb_freeA_gen = np.std(fold_results_cftb_freeA_gen, axis=0)

    std_cftb_freeB_accA = np.std(fold_results_cftb_freeB_accA, axis=0)
    std_cftb_freeB_accB = np.std(fold_results_cftb_freeB_accB, axis=0)
    std_cftb_freeB_gen = np.std(fold_results_cftb_freeB_gen, axis=0)

    std_cftb_busyA_accA = np.std(fold_results_cftb_busyA_accA, axis=0)
    std_cftb_busyA_accB = np.std(fold_results_cftb_busyA_accB, axis=0)
    std_cftb_busyA_gen = np.std(fold_results_cftb_busyA_gen, axis=0)

    std_cftb_busyB_accA = np.std(fold_results_cftb_busyB_accA, axis=0)
    std_cftb_busyB_accB = np.std(fold_results_cftb_busyB_accB, axis=0)
    std_cftb_busyB_gen = np.std(fold_results_cftb_busyB_gen, axis=0)

    std_cftb_specB_accA = np.std(fold_results_cftb_specB_accA, axis=0)
    std_cftb_specB_accB = np.std(fold_results_cftb_specB_accB, axis=0)
    std_cftb_specB_gen = np.std(fold_results_cftb_specB_gen, axis=0)

    std_cftb_rnd_accA = np.std(fold_results_cftb_rnd_accA, axis=0)
    std_cftb_rnd_accB = np.std(fold_results_cftb_rnd_accB, axis=0)
    std_cftb_rnd_gen = np.std(fold_results_cftb_rnd_gen, axis=0)

    #fig, axs = plt.subplots(1, 1, figsize=(12, 6))
    # Plot - Accuracy of Previous Knowledge
    plt.figure(figsize=(12, 6))
    plt.axhline(y=ftb_accA, color='dimgrey', linestyle='solid', linewidth=1.5, alpha=0.80, label='Full FT')
    plt.axhline(y=Loraftb_accA, color='crimson', linestyle=(0, (5, 2, 1, 2)), linewidth=2.5, alpha=0.80, label='LoRA')


    # Creating custom x labels
    x_labels = list(cftb_results.keys())

    # First plotl 
    plt.errorbar(x_labels, cftb_busyB_accA, yerr=std_cftb_busyB_accA, **strategy_map_short['busy_B'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_freeA_accA, yerr=std_cftb_freeA_accA, **strategy_map_short['free_A'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_specB_accA, yerr=std_cftb_specB_accA, **strategy_map_short['spec_B'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_rnd_accA, yerr=std_cftb_rnd_accA, **strategy_map_short['rnd'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.xlabel('Number of updated neurons')
    plt.ylabel('Accuracy')
    #axs.set_title('Sequential learning experiment - Accuracy of Previous Knowledge')
    #plt.legend(loc=loc_old, bbox_to_anchor=bbox_old)
    plt.grid(True)
    plt.ylim(y_lim_old_1, y_lim_old_2)
    plt.tight_layout()
    plt.xticks(ticks=x_labels, labels=[f"{int(int(x)/1000)}k" for x in x_labels])
    plt.savefig(f"{output_dir}/{filename[:-5]}_neuron_update_strategies_old_knowledge.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    # Plot - Accuracy of New Knowledge
    plt.figure(figsize=(12, 6))
    plt.axhline(y=ftb_accB, color='dimgrey', linestyle='solid', linewidth=1.5, alpha=0.80, label='Full FT')
    plt.axhline(y=Loraftb_accB, color='crimson', linestyle=(0, (5, 2, 1, 2)), linewidth=2.5, alpha=0.80, label='LoRA')
    plt.errorbar(x_labels, cftb_busyB_accB, yerr=std_cftb_busyB_accB, **strategy_map_short['busy_B'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_freeA_accB, yerr=std_cftb_freeA_accB, **strategy_map_short['free_A'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_specB_accB, yerr=std_cftb_specB_accB, **strategy_map_short['spec_B'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_rnd_accB, yerr=std_cftb_rnd_accB, **strategy_map_short['rnd'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.xlabel('Number of updated neurons')
    plt.ylabel('Accuracy')
    plt.legend(loc=loc_new, bbox_to_anchor=bbox_new)
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(ticks=x_labels, labels=[f"{int(int(x)/1000)}k" for x in x_labels])
    plt.savefig(f"{output_dir}/{filename[:-5]}_neuron_update_strategies_new_knowledge.pdf", format='pdf', bbox_inches='tight')
    plt.show()


    #plt.figure(figsize=(12, 6))
    plt.figure(figsize=(12, 6))

    plt.axhline(y=ftb_accB, color='dimgrey', linestyle='solid', linewidth=1.5, alpha=0.80, label='Full FT')

    plt.errorbar(x_labels, cftb_busyB_gen, yerr=std_cftb_busyB_gen, **strategy_map_short['busy_B'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_freeA_gen, yerr=std_cftb_freeA_gen, **strategy_map_short['free_A'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_specB_gen, yerr=std_cftb_specB_gen, **strategy_map_short['spec_B'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_rnd_gen, yerr=std_cftb_rnd_gen, **strategy_map_short['rnd'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)

    plt.xlabel('Number of updated neurons')
    plt.ylabel('Accuracy')
    #plt.legend(loc=loc_gen, bbox_to_anchor=bbox_gen)
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(ticks=x_labels, labels=[f"{int(int(x)/1000)}k" for x in x_labels])
    plt.savefig(f"{output_dir}/{filename[:-5]}_neuron_update_strategies_general_knowledge.pdf", format='pdf', bbox_inches='tight')
    plt.show()


def plot_editing_old_new_general_noLoRa(experiment_name, filename, filepath, strategies, loc_old, bbox_old,loc_new, bbox_new, loc_gen, bbox_gen, y_lim_old_1, y_lim_old_2):
    # Set up the plot style
    # plt.style.use('seaborn-whitegrid')
    # plt.rcParams.update({
    #     'font.size': 18,
    #     'axes.labelsize': 20,
    #     'axes.titlesize': 18,
    #     'xtick.labelsize': 16,
    #     'ytick.labelsize': 16,
    #     'legend.fontsize': 16,
    #     'font.family': 'serif',
    #     'text.usetex': True,
    #     'figure.figsize': (12, 6)
    # })
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'font.family': 'serif',
        'text.usetex': True,
        'figure.figsize': (10, 8)
    })
    # Create output directory if it doesn't exist
    output_dir = f"./figures/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    # Load data
    with open(os.path.join(filepath, filename), 'r') as file: 
        data = json.load(file)
    # with open(f'../results/experiment_3_1/experiment_3_1_2000_{N_SAMPLES}.json', 'r') as file: 
    #     data = json.load(file)

    results = data['results']

    n_folds =  len(data['results'].keys())
    n_thresh = len(list(data['results'].values())[0]['cft-notb'].keys())

    fta_accA = 0

    ftb_accA = 0
    ftb_accB = 0
    ftb_acc_gen = 0

    # Arrays to store individual fold results for calculating std dev
    fold_results_cftb_busyA_accA = np.zeros((n_folds, n_thresh))
    fold_results_cftb_busyA_accB = np.zeros((n_folds, n_thresh))
    fold_results_cftb_busyA_gen = np.zeros((n_folds, n_thresh))

    fold_results_cftb_freeA_accA = np.zeros((n_folds, n_thresh))
    fold_results_cftb_freeA_accB = np.zeros((n_folds, n_thresh))
    fold_results_cftb_freeA_gen = np.zeros((n_folds, n_thresh))

    fold_results_cftb_busyB_accA = np.zeros((n_folds, n_thresh))
    fold_results_cftb_busyB_accB = np.zeros((n_folds, n_thresh))
    fold_results_cftb_busyB_gen = np.zeros((n_folds, n_thresh))

    fold_results_cftb_freeB_accA = np.zeros((n_folds, n_thresh))
    fold_results_cftb_freeB_accB = np.zeros((n_folds, n_thresh))
    fold_results_cftb_freeB_gen = np.zeros((n_folds, n_thresh))

    fold_results_cftb_specB_accA = np.zeros((n_folds, n_thresh))
    fold_results_cftb_specB_accB = np.zeros((n_folds, n_thresh))
    fold_results_cftb_specB_gen = np.zeros((n_folds, n_thresh))

    fold_results_cftb_rnd_accA = np.zeros((n_folds, n_thresh))
    fold_results_cftb_rnd_accB = np.zeros((n_folds, n_thresh))
    fold_results_cftb_rnd_gen = np.zeros((n_folds, n_thresh))

    for i, fold in enumerate(data['results']):

        ftb_accA += data['results'][fold]['ft-notb']['avg_accA']
        ftb_accB += data['results'][fold]['ft-notb']['avg_accB']

        cftb_results = data['results'][fold]['cft-notb']

        fold_results_cftb_busyA_accA[i, :] = np.array([cftb_results[t]['busy_A']['avg_accA'] for t in cftb_results]).astype(float)
        fold_results_cftb_busyA_accB[i, :] = np.array([cftb_results[t]['busy_A']['avg_accB'] for t in cftb_results]).astype(float)
        fold_results_cftb_busyA_gen[i, :] = np.array([cftb_results[t]['busy_A']['avg_acc_gen'] for t in cftb_results]).astype(float)

        fold_results_cftb_busyB_accA[i, :] = np.array([cftb_results[t]['busy_B']['avg_accA'] for t in cftb_results]).astype(float)
        fold_results_cftb_busyB_accB[i, :] = np.array([cftb_results[t]['busy_B']['avg_accB'] for t in cftb_results]).astype(float)
        fold_results_cftb_busyB_gen[i, :] = np.array([cftb_results[t]['busy_B']['avg_acc_gen'] for t in cftb_results]).astype(float)

        fold_results_cftb_freeA_accA[i, :] = np.array([cftb_results[t]['free_A']['avg_accA'] for t in cftb_results]).astype(float)
        fold_results_cftb_freeA_accB[i, :] = np.array([cftb_results[t]['free_A']['avg_accB'] for t in cftb_results]).astype(float)
        fold_results_cftb_freeA_gen[i, :] = np.array([cftb_results[t]['free_A']['avg_acc_gen'] for t in cftb_results]).astype(float)

        fold_results_cftb_freeB_accA[i, :] = np.array([cftb_results[t]['free_B']['avg_accA'] for t in cftb_results]).astype(float)
        fold_results_cftb_freeB_accB[i, :] = np.array([cftb_results[t]['free_B']['avg_accB'] for t in cftb_results]).astype(float)
        fold_results_cftb_freeB_gen[i, :] = np.array([cftb_results[t]['free_B']['avg_acc_gen'] for t in cftb_results]).astype(float)

        fold_results_cftb_specB_accA[i, :] = np.array([cftb_results[t]['spec_B']['avg_accA'] for t in cftb_results]).astype(float)
        fold_results_cftb_specB_accB[i, :] = np.array([cftb_results[t]['spec_B']['avg_accB'] for t in cftb_results]).astype(float)
        fold_results_cftb_specB_gen[i, :] = np.array([cftb_results[t]['spec_B']['avg_acc_gen'] for t in cftb_results]).astype(float)

        fold_results_cftb_rnd_accA[i, :] = np.array([cftb_results[t]['rnd']['avg_accA'] for t in cftb_results]).astype(float)
        fold_results_cftb_rnd_accB[i, :] = np.array([cftb_results[t]['rnd']['avg_accB'] for t in cftb_results]).astype(float)
        fold_results_cftb_rnd_gen[i, :] = np.array([cftb_results[t]['rnd']['avg_acc_gen'] for t in cftb_results]).astype(float)

    fta_accA /= n_folds
    ftb_accA /= n_folds
    ftb_accB /= n_folds

    cftb_freeA_accA = np.mean(fold_results_cftb_freeA_accA, axis=0)
    cftb_freeA_accB = np.mean(fold_results_cftb_freeA_accB, axis=0)
    cftb_freeA_gen = np.mean(fold_results_cftb_freeA_gen, axis=0)

    cftb_freeB_accA = np.mean(fold_results_cftb_freeB_accA, axis=0)
    cftb_freeB_accB = np.mean(fold_results_cftb_freeB_accB, axis=0)
    cftb_freeB_gen = np.mean(fold_results_cftb_freeB_gen, axis=0)

    cftb_busyA_accA = np.mean(fold_results_cftb_busyA_accA, axis=0)
    cftb_busyA_accB = np.mean(fold_results_cftb_busyA_accB, axis=0)
    cftb_busyA_gen = np.mean(fold_results_cftb_busyA_gen, axis=0)

    cftb_busyB_accA = np.mean(fold_results_cftb_busyB_accA, axis=0)
    cftb_busyB_accB = np.mean(fold_results_cftb_busyB_accB, axis=0)
    cftb_busyB_gen = np.mean(fold_results_cftb_busyB_gen, axis=0)

    cftb_specB_accA = np.mean(fold_results_cftb_specB_accA, axis=0)
    cftb_specB_accB = np.mean(fold_results_cftb_specB_accB, axis=0)
    cftb_specB_gen = np.mean(fold_results_cftb_specB_gen, axis=0)

    cftb_rnd_accA = np.mean(fold_results_cftb_rnd_accA, axis=0)
    cftb_rnd_accB = np.mean(fold_results_cftb_rnd_accB, axis=0)
    cftb_rnd_gen = np.mean(fold_results_cftb_rnd_gen, axis=0)

    # Calculate standard deviation
    std_cftb_freeA_accA = np.std(fold_results_cftb_freeA_accA, axis=0)
    std_cftb_freeA_accB = np.std(fold_results_cftb_freeA_accB, axis=0)
    std_cftb_freeA_gen = np.std(fold_results_cftb_freeA_gen, axis=0)

    std_cftb_freeB_accA = np.std(fold_results_cftb_freeB_accA, axis=0)
    std_cftb_freeB_accB = np.std(fold_results_cftb_freeB_accB, axis=0)
    std_cftb_freeB_gen = np.std(fold_results_cftb_freeB_gen, axis=0)

    std_cftb_busyA_accA = np.std(fold_results_cftb_busyA_accA, axis=0)
    std_cftb_busyA_accB = np.std(fold_results_cftb_busyA_accB, axis=0)
    std_cftb_busyA_gen = np.std(fold_results_cftb_busyA_gen, axis=0)

    std_cftb_busyB_accA = np.std(fold_results_cftb_busyB_accA, axis=0)
    std_cftb_busyB_accB = np.std(fold_results_cftb_busyB_accB, axis=0)
    std_cftb_busyB_gen = np.std(fold_results_cftb_busyB_gen, axis=0)

    std_cftb_specB_accA = np.std(fold_results_cftb_specB_accA, axis=0)
    std_cftb_specB_accB = np.std(fold_results_cftb_specB_accB, axis=0)
    std_cftb_specB_gen = np.std(fold_results_cftb_specB_gen, axis=0)

    std_cftb_rnd_accA = np.std(fold_results_cftb_rnd_accA, axis=0)
    std_cftb_rnd_accB = np.std(fold_results_cftb_rnd_accB, axis=0)
    std_cftb_rnd_gen = np.std(fold_results_cftb_rnd_gen, axis=0)

    #fig, axs = plt.subplots(1, 1, figsize=(12, 6))
    # Plot - Accuracy of Previous Knowledge
    plt.figure(figsize=(12, 6))
    plt.axhline(y=ftb_accA, color='dimgrey', linestyle='solid', linewidth=1.5, alpha=0.80, label='Full FT')


    # Creating custom x labels
    x_labels = list(cftb_results.keys())

    # First plotl 
    plt.errorbar(x_labels, cftb_busyB_accA, yerr=std_cftb_busyB_accA, **strategy_map_short['busy_B'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_freeA_accA, yerr=std_cftb_freeA_accA, **strategy_map_short['free_A'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_specB_accA, yerr=std_cftb_specB_accA, **strategy_map_short['spec_B'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_rnd_accA, yerr=std_cftb_rnd_accA, **strategy_map_short['rnd'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.xlabel('Number of updated neurons')
    plt.ylabel('Accuracy')
    #axs.set_title('Sequential learning experiment - Accuracy of Previous Knowledge')
    plt.legend(loc=loc_old, bbox_to_anchor=bbox_old)
    plt.grid(True)
    plt.ylim(y_lim_old_1, y_lim_old_2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename[:-5]}_neuron_update_strategies_old_knowledge.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    # Plot - Accuracy of New Knowledge
    plt.figure(figsize=(12, 6))
    plt.axhline(y=ftb_accB, color='dimgrey', linestyle='solid', linewidth=1.5, alpha=0.80, label='Full FT')
    plt.errorbar(x_labels, cftb_busyB_accB, yerr=std_cftb_busyB_accB, **strategy_map_short['busy_B'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_freeA_accB, yerr=std_cftb_freeA_accB, **strategy_map_short['free_A'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_specB_accB, yerr=std_cftb_specB_accB, **strategy_map_short['spec_B'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_rnd_accB, yerr=std_cftb_rnd_accB, **strategy_map_short['rnd'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.xlabel('Number of updated neurons')
    plt.ylabel('Accuracy')
    plt.legend(loc=loc_new, bbox_to_anchor=bbox_new)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename[:-5]}_neuron_update_strategies_new_knowledge.pdf", format='pdf', bbox_inches='tight')
    plt.show()


    #plt.figure(figsize=(12, 6))
    plt.figure(figsize=(12, 6))

    plt.axhline(y=ftb_accB, color='dimgrey', linestyle='solid', linewidth=1.5, alpha=0.80, label='Full FT')

    plt.errorbar(x_labels, cftb_busyB_gen, yerr=std_cftb_busyB_gen, **strategy_map_short['busy_B'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_freeA_gen, yerr=std_cftb_freeA_gen, **strategy_map_short['free_A'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_specB_gen, yerr=std_cftb_specB_gen, **strategy_map_short['spec_B'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)
    plt.errorbar(x_labels, cftb_rnd_gen, yerr=std_cftb_rnd_gen, **strategy_map_short['rnd'], linewidth=1.5, markersize=10, capsize=5, elinewidth=1.5, capthick=1.5)

    plt.xlabel('Number of updated neurons')
    plt.ylabel('Accuracy')
    plt.legend(loc=loc_gen, bbox_to_anchor=bbox_gen)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename[:-5]}_neuron_update_strategies_general_knowledge.pdf", format='pdf', bbox_inches='tight')
    plt.show()
################ Old Plots ###################""
    
    # Function to plot old and new knowledge accuracies
def plot_new_knowledge_all_long_legend(filepath, filename, experiment_name, strategies=None, loc_new='lower right', bbox_new=(1, 1)):
    # Set up the plot style
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 26,
        'axes.titlesize': 24,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 22,
        'font.family': 'serif',
        'text.usetex': True,
        'figure.figsize': (10, 8)
    })

    # Load data
    with open(os.path.join(filepath, filename), 'r') as file: 
        data = json.load(file)
    
    # Create output directory if it doesn't exist
    output_dir = f"./figures/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
 
 
    all_strategies = list(strategy_map.keys())

    if strategies is None:
        strategies = all_strategies

    n_folds =  len(data['results'].keys())
    n_thresh = len(list(data['results'].values())[0]['cfta'].keys())


    # Initialize accumulators for standard deviation calculations
    cftb_stub_accB_list = np.zeros((n_folds, n_thresh))
    cftb_free_accB_list = np.zeros((n_folds, n_thresh))
    cftb_rnd_accB_list = np.zeros((n_folds, n_thresh))

    ftb_accB = 0

    for i, fold in enumerate(data['results']):
        ftb_accB += data['results'][fold]['fta']['acc_A']

        cftb_results = data['results'][fold]['cfta']

        cftb_stub_accB_list[i, :] = np.array([cftb_results[t]['busy_H']['avg_accA'] for t in cftb_results]).astype(float)
        cftb_free_accB_list[i, :] = np.array([cftb_results[t]['free_H']['avg_accA'] for t in cftb_results]).astype(float)
        cftb_rnd_accB_list[i, :] = np.array([cftb_results[t]['rnd']['avg_accA'] for t in cftb_results]).astype(float)

    ftb_accB /= n_folds
    cftb_stub_accB = np.mean(cftb_stub_accB_list, axis=0)
    cftb_free_accB = np.mean(cftb_free_accB_list, axis=0)
    cftb_rnd_accB = np.mean(cftb_rnd_accB_list, axis=0)

    # Calculate standard deviations
    cftb_stub_accB_std = np.std(cftb_stub_accB_list, axis=0)
    cftb_free_accB_std = np.std(cftb_free_accB_list, axis=0)
    cftb_rnd_accB_std = np.std(cftb_rnd_accB_list, axis=0)

    # Creating custom x labels
    x_labels = list(cftb_results.keys())


    # Plot - Accuracy of New Knowledge
    plt.figure(figsize=(10, 6))
    plt.axhline(y=ftb_accB, color='dimgrey', linestyle='solid', linewidth=1.5, alpha=0.80, label='Full FT')
    
    plt.errorbar(x_labels, cftb_free_accB, yerr=cftb_free_accB_std, linestyle=strategy_map['free_H']['linestyle'], linewidth=1.5, color=strategy_map['free_H']['color'], marker=strategy_map['free_H']['marker'], 
                 markersize=10, capsize=5, elinewidth=1.5, capthick=1.5, label=strategy_map['free_H']['label'])
    plt.errorbar(x_labels, cftb_stub_accB, yerr=cftb_stub_accB_std, linestyle=strategy_map['busy_H']['linestyle'], linewidth=1.5, color=strategy_map['busy_H']['color'], marker=strategy_map['busy_H']['marker'], 
                 markersize=10, capsize=5, elinewidth=1.5, capthick=1.5, label=strategy_map['busy_H']['label'])
    plt.errorbar(x_labels, cftb_rnd_accB, yerr=cftb_rnd_accB_std, linestyle=strategy_map['rnd']['linestyle'], linewidth=1.5, color=strategy_map['rnd']['color'], marker=strategy_map['rnd']['marker'], 
                 markersize=10, capsize=5, elinewidth=1.5, capthick=1.5, label=strategy_map['rnd']['label'])


    plt.xlabel('Number of updated neurons')
    plt.ylabel('Accuracy')
    plt.legend(loc=loc_new, bbox_to_anchor=bbox_new)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename[:-5]}_neuron_update_strategies_new_knowledge.pdf", format='pdf', bbox_inches='tight')
    plt.show()


# Function to plot old and new knowledge accuracies
def plot_old_new_knowledge_all_v0(filepath, filename, experiment_name, strategies=None, loc_old='lower left', bbox_old=(1, 1), loc_new='lower right', bbox_new=(1, 1), y_lim_old_1=0.8, y_lim_old_2 = 1.01):
    # Set up the plot style
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 20,
        'axes.titlesize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'font.family': 'serif',
        'text.usetex': True,
        'figure.figsize': (10, 8)
    })

    # Load data
    with open(os.path.join(filepath, filename), 'r') as file: 
        data = json.load(file)
    
    # Create output directory if it doesn't exist
    output_dir = f"./figures/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)

    # This is likely overriding the plt parameters :(
    #sns.set_theme(style="whitegrid", context="paper", font_scale=1.6)
 
 
    all_strategies = list(strategy_map.keys())

    if strategies is None:
        strategies = all_strategies

    # Initialize fold results only for the selected strategies
    n_folds = len(data['results'].keys())
    n_thresh = len(list(data['results'].values())[0]['cftb'].keys())
    
    fold_results = {strategy: {'accA': np.zeros((n_folds, n_thresh)), 'accB': np.zeros((n_folds, n_thresh))} for strategy in strategies}

    fta_accA = 0
    ftb_accA = 0
    ftb_accB = 0

    for i, fold in enumerate(data['results']):
        fta_accA += data['results'][fold]['fta']['acc_A']
        ftb_accA += data['results'][fold]['ftb']['avg_accA']
        ftb_accB += data['results'][fold]['ftb']['avg_accB']

        cftb_results = data['results'][fold]['cftb']

        for strategy in strategies:
            fold_results[strategy]['accA'][i, :] = np.array([cftb_results[t][strategy]['avg_accA'] for t in cftb_results]).astype(float)
            fold_results[strategy]['accB'][i, :] = np.array([cftb_results[t][strategy]['avg_accB'] for t in cftb_results]).astype(float)

    fta_accA /= n_folds
    ftb_accA /= n_folds
    ftb_accB /= n_folds

    # Calculate means and standard deviations
    means = {strategy: {'accA': np.mean(fold_results[strategy]['accA'], axis=0), 'accB': np.mean(fold_results[strategy]['accB'], axis=0)} for strategy in strategies}
    stds = {strategy: {'accA': np.std(fold_results[strategy]['accA'], axis=0), 'accB': np.std(fold_results[strategy]['accB'], axis=0)} for strategy in strategies}

    # Creating custom x labels
    x_labels = list(cftb_results.keys())

    # Plot - Accuracy of Previous Knowledge
    plt.figure(figsize=(10, 6))
    plt.axhline(y=fta_accA, color='firebrick', linestyle='solid', linewidth=1.5, alpha=0.80, label='Initial')
    plt.axhline(y=ftb_accA, color='dimgrey', linestyle='solid', linewidth=1.5, alpha=0.80, label='Full FT')
    
    for strategy in strategies:
        plt.errorbar(x_labels, means[strategy]['accA'], yerr=stds[strategy]['accA'], 
                     linestyle=strategy_map[strategy]['linestyle'], linewidth=1.5, 
                     color=strategy_map[strategy]['color'], marker=strategy_map[strategy]['marker'], 
                     markersize=10, capsize=5, elinewidth=1.5, capthick=1.5, label=strategy_map[strategy]['label'])

    plt.xlabel('Number of updated neurons')
    plt.ylabel('Accuracy')
    plt.legend(loc=loc_old, bbox_to_anchor=bbox_old)
    plt.grid(True)
    plt.ylim(y_lim_old_1,y_lim_old_2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename[:-5]}_neuron_update_strategies_old_knowledge.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    # Plot - Accuracy of New Knowledge
    plt.figure(figsize=(10, 6))
    plt.axhline(y=ftb_accB, color='dimgrey', linestyle='solid', linewidth=1.5, alpha=0.80, label='Full FT')
    
    for strategy in strategies:
        plt.errorbar(x_labels, means[strategy]['accB'], yerr=stds[strategy]['accB'], 
                     linestyle=strategy_map[strategy]['linestyle'], linewidth=1.5, 
                     color=strategy_map[strategy]['color'], marker=strategy_map[strategy]['marker'], 
                     markersize=10, capsize=5, elinewidth=1.5, capthick=1.5, label=strategy_map[strategy]['label'])

    plt.xlabel('Number of updated neurons')
    plt.ylabel('Accuracy')
    plt.legend(loc=loc_new, bbox_to_anchor=bbox_new)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename[:-5]}_neuron_update_strategies_new_knowledge.pdf", format='pdf', bbox_inches='tight')
    plt.show()