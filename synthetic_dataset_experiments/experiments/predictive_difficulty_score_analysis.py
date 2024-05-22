
"""
Code used to generate the plot showing the standard deviation of the performance scores with respect to the VarZt
(Figure 12 of the paper)
"""

# Script to generate all the plots with respect to the predictive difficulty scoreßß
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import viridis
import seaborn as sns
import pickle

def legend_title_left(leg):
                c = leg.get_children()[0]
                title = c.get_children()[0]
                hpack = c.get_children()[1]
                c._children = [hpack]
                hpack._children = [title] + hpack.get_children()


dataset_dnn_stochasticity_dict = {
    "sameInit_1": {
        "d-convlstm": {
            100: "/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/evaluator/evalSameInit_pyramid_v2_stochastic_multiscale_ablation__d72_p100_train_1_eval_1",
            95: "/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/evaluator/evalSameInit_pyramid_v2_stochastic_multiscale_ablation__d72_p95_train_1_eval_1",
            90: "/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/evaluator/evalSameInit_pyramid_v2_stochastic_multiscale_ablation__d72_p90_train_1_eval_1",
            85: "/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/evaluator/evalSameInit_pyramid_v2_stochastic_multiscale_ablation__d72_p85_train_1_eval_1",
            80: "/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/evaluator/evalSameInit_pyramid_v2_stochastic_multiscale_ablation__d72_p80_train_1_eval_1"
        },
    },
    
    "diffInit": {
        "d-convlstm": {
            100: "/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/evaluator/evalDiffInit_pyramid_v2_stochastic_multiscale_ablation__d72_p100_train_1_eval_1",
            95: "/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/evaluator/evalDiffInit_pyramid_v2_stochastic_multiscale_ablation__d72_p95_train_1_eval_1",
            90: "/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/evaluator/evalDiffInit_pyramid_v2_stochastic_multiscale_ablation__d72_p90_train_1_eval_1",
            85: "/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/evaluator/evalDiffInit_pyramid_v2_stochastic_multiscale_ablation__d72_p85_train_1_eval_1",
            80: "/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/evaluator/evalDiffInit_pyramid_v2_stochastic_multiscale_ablation__d72_p80_train_1_eval_1"
        }
    },
}

class SD_weight_generator:
    @staticmethod
    def load_temporal_dos(loadpath, scales, stochasticity_list, max_time_step=None):
        """
        Load the temporal DOS distribution and compute the SD vs time step
        Args:
            loadpath: Path to the pickle file
            scales: List of scales
            stochasticity_list: List of stochasticity values
            max_time_step: Maximum time step to consider
            
        Returns:
            SD_vs_time_step_dict: A dictionary containing the SD vs time step for each scale and stochasticity
                    Format: {scale: {stochasticity: {time_step: sd_value}}} e.g. {1: {100: {0: 0.0, 1: 0.0, 2: 0.0, ...}, 95: {0: 0.0, 1: 0.0, 2: 0.0, ...}, ...}, ...}
        """    
        SD_vs_time_step_dict = {}
        
        with open(loadpath, 'rb') as handle:
            sweepVal_vs_num_unburnt_trees_dict = pickle.load(handle)

        for scale in scales:
            SD_vs_time_step_dict[scale] = {}
            for idx, stoch_val in enumerate(stochasticity_list):
                num_unburnt_trees_dict = sweepVal_vs_num_unburnt_trees_dict[stoch_val]
                stoch_val = int(stoch_val * 100)
                SD_vs_time_step_dict[scale][stoch_val] = {}
                
                
                if max_time_step is None:
                    time_step_list = sorted(num_unburnt_trees_dict[0].keys())
                else:
                    time_step_list = range(max_time_step)
                
                for time_step in sorted(time_step_list):
                    values_at_time = np.array([run_dict[time_step][scale] for run_dict in num_unburnt_trees_dict.values() if time_step in run_dict.keys()])
                    sd_value = np.std(values_at_time)
                    SD_vs_time_step_dict[scale][stoch_val][time_step] = sd_value
                
        return SD_vs_time_step_dict
    

def load_SD_vs_time_step_dict():
    """
    SD_vs_time_step_dict: A dictionary containing the SD vs time step for each scale and stochasticity
    Format: {stochasticity: {time_step: SD, ...}, ...}
    """
    # Load the temporal DOS distribution from the specified pickle file
    pickle_path = "../NetLogo-Synthetic-Dataset-Generator/res/temporal_300_stochasticity_vs_num_unburnt_trees_dict_64grid_same_fire_12345_same_forest_47822.pickle"
    SD_vs_time_step_dict = SD_weight_generator.load_temporal_dos(loadpath=pickle_path,
                                                                scales=[1],
                                                                stochasticity_list=[1, 0.95, 0.90, 0.85, 0.80],
                                                                max_time_step=60)
    SD_vs_time_step_dict = SD_vs_time_step_dict[1] # Only consider the scale 1
    return SD_vs_time_step_dict
    
def load_raw_metric_scores(dnn_name, stochasticity, dataset_nameType, metric_name):
    """
    Return the raw performance scores for a given DNN and stochasticity.
    format: {time_step: [list of scores across all the evaluation chunks], ...}
    """
    # Load the raw performance scores from the dnn
    base_folder_name = dataset_dnn_stochasticity_dict[dataset_nameType][dnn_name][stochasticity]
    metric_json_path = os.path.join(base_folder_name, "raw_metric_data.json")
    with open(metric_json_path) as f:
        metric_data = json.load(f)
    scale = "1"
    
    print(stochasticity, dnn_name, metric_data.keys())
    
    filtered_metric_data = metric_data[metric_name][scale] # Format: {time_step: [list of scores across all the evaluation chunks], ...}
    return filtered_metric_data

def generate_dataframe(dnn_name_list, stochasticity_list, dataset_nameType, metric_name_list, savepath_dataframes=None):
    # Goal is to create a data frame where each row is a time instant and these columns: SD, time_step, S-Level, MSE, AUC-PR, F1, Recall, DNN
    consolidated_dataframes = []
    
    # Load the SD vs time step dictionary
    SD_vs_time_step_dict = load_SD_vs_time_step_dict()
    
    # Load the raw scores for each DNN
    for dnn_name in dnn_name_list:
        for stochasticity in stochasticity_list:
            for metric_name in metric_name_list:
                # Load the raw performance scores for the dnn for the given stochasticity
                metric_data = load_raw_metric_scores(dnn_name, stochasticity, dataset_nameType, metric_name)
                
                for timestep, score_list_t in metric_data.items():
                    if int(timestep) > 9: # Ignore the first 10 time steps as they are in the observation phase
                        
                        for score in score_list_t:
                            SD = SD_vs_time_step_dict[stochasticity][int(timestep)]
                            consolidated_dataframes.append([SD, int(timestep), stochasticity, score, metric_name, dnn_name])
                
    # Create a dataframe from the consolidated data
    df = pd.DataFrame(consolidated_dataframes, columns=["SD", "time_step", "S-Level", "score", "metric-name", "DNN"])

    if savepath_dataframes is not None:
        df.to_csv(savepath_dataframes, index=False)


class visualizer:    
    @staticmethod
    def generate_metricSD_vs_varzt_line_plots(dataframe_path, savepath_plots, dnn_name, s_level, normalize=False):
        df = pd.read_csv(dataframe_path)
        os.makedirs(savepath_plots, exist_ok=True)
        
        # Filter the DataFrame by DNN name and S-Level
        df_filtered = df[(df['DNN'] == dnn_name) & (df['S-Level'] == s_level) & (df['time_step'] >= 10)]

        # Set a professional color palette
        colors = plt.get_cmap('tab10')

        # Prepare the plot with a professional style
        plt.style.use('classic')
        plt.figure(figsize=(8, 5))

        # Define markers and create an iterator
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'd', '|', '_']
        marker_iter = iter(markers)

        # Plot each metric with a different color and marker
        for i, metric in enumerate(df_filtered['metric-name'].unique()):
            if metric == "F1":
                continue
            df_metric = df_filtered[df_filtered['metric-name'] == metric]

            # Compute standard deviation of scores for each SD value
            sd_std_scores = df_metric.groupby('SD')['score'].std().reset_index()

            # Normalize the standard deviation if flag is True
            if normalize:
                max_std = sd_std_scores['score'].max()
                sd_std_scores['score'] = sd_std_scores['score'] / max_std

            # Add a line for this metric to the plot with different markers
            if metric == "ROC_PR":
                metric_name = "AUC-PR"
            else:
                metric_name = metric
            
            plt.plot(sd_std_scores['SD'], sd_std_scores['score'], marker=next(marker_iter), linestyle='--', color=colors(i),
                    label=metric_name, markersize=8, zorder=3)

        # Configure plot aesthetics
        # plt.title(f'{"Normalized " if normalize else ""}Standard Deviation for Various Metrics', fontsize=14)
        plt.xlabel('SD', fontsize=18, fontweight='bold')
        plt.ylabel('Normalized Standard Deviation of Score' if normalize else 'Standard Deviation of Metric', fontsize=18, fontweight='bold')
        
        # Set grid with transparency (alpha) and behind plot elements (zorder)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.2, zorder=1)
        
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        # Create a legend with the title 'Metric' and make it top left
        legend = plt.legend(title='Metric', fontsize=10, title_fontsize=12, ncol=4, loc='upper left')
        plt.setp(legend.get_texts(), fontsize=10)
        legend_title_left(legend)

        # Save the plot
        plot_filename = os.path.join(savepath_plots, f'{"normalized_" if normalize else ""}combined_line_plot_S-Level_{s_level}_{dnn_name}.pdf')
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        plt.close()

    
def main():
    dnn_name_list = ["d-convlstm"]
    
    dataset_nameType="sameInit_1" # "sameInit_1" "diffInit"
    metric_name_list = ["MSE", "ROC_PR", "Recall", "Precision"]
    stochasticity_list = [100, 95, 90, 85, 80]
    
    base_savepath = "Enter the base path to save the dataframe and the line plots"
    os.makedirs(base_savepath, exist_ok=True)
    fpath = os.path.join(base_savepath, "dataframe.csv")
    
    # Goal is to create a data frame where each row is a time instant and these columns: SD, time_step, S-Level, MSE, AUC-PR, F1, Recall, DNN
    generate_dataframe(dnn_name_list, stochasticity_list, dataset_nameType, metric_name_list, savepath_dataframes=fpath)
    
    base_savepath = "Enter the base path to save the plots"
    plots_savepath = f"{base_savepath}/metricSD_vs_varzt/"
    visualizer.generate_metricSD_vs_varzt_line_plots(dataframe_path = fpath,
                                                    savepath_plots=plots_savepath,
                                                    dnn_name=dnn_name_list[0],
                                                    s_level=80,
                                                    normalize=False)
    
    
if __name__ == "__main__":
    main()