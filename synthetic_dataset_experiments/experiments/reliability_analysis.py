"""
Code used to generate the results that support the claim that DNNs are learning to 
predict the statistic-GT (Figure 10 and Figure 11).
"""

import pickle
from predictive_difficulty_score_analysis import load_SD_vs_time_step_dict
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np



def load_raw_pred_forecastGT_dict(dnn_name, stochasticity_level, dataset_type="sameInit", base_loadpath=None):
    """
    Load the raw prediction and forecastGT data for the given DNN and stochasticity level
    """
    raw_pred_forecastGT_dict = {
        "sameInit": {
            ############################################## trained DNNs tested on ESP of same S-Level ##############################################
            "d-convlstm": {
                    100: f"{base_loadpath}/evalSameInit_pyramid_v2_stochastic_multiscale_ablation__d72_p100_train_1_eval_1/raw_prediction_forecastGT.pkl",
                    95: f"{base_loadpath}/evalSameInit_pyramid_v2_stochastic_multiscale_ablation__d72_p95_train_1_eval_1/raw_prediction_forecastGT.pkl",
                    90: f"{base_loadpath}/evalSameInit_pyramid_v2_stochastic_multiscale_ablation__d72_p90_train_1_eval_1/raw_prediction_forecastGT.pkl",
                    85: f"{base_loadpath}/evalSameInit_pyramid_v2_stochastic_multiscale_ablation__d72_p85_train_1_eval_1/raw_prediction_forecastGT.pkl",
                    80: f"{base_loadpath}/evalSameInit_pyramid_v2_stochastic_multiscale_ablation__d72_p80_train_1_eval_1/raw_prediction_forecastGT.pkl"
                }
        },
        "diffInit": {
            ############################################## trained DNNs tested on same S-Level ##############################################
            "d-convlstm": {
                    100: f"{base_loadpath}/evalDiffInit_pyramid_v2_stochastic_multiscale_ablation__d72_p100_train_1_eval_1/raw_prediction_forecastGT.pkl",
                    95: f"{base_loadpath}/evalDiffInit_pyramid_v2_stochastic_multiscale_ablation__d72_p95_train_1_eval_1/raw_prediction_forecastGT.pkl",
                    90: f"{base_loadpath}/evalDiffInit_pyramid_v2_stochastic_multiscale_ablation__d72_p90_train_1_eval_1/raw_prediction_forecastGT.pkl",
                    85: f"{base_loadpath}/evalDiffInit_pyramid_v2_stochastic_multiscale_ablation__d72_p85_train_1_eval_1/raw_prediction_forecastGT.pkl",
                    80: f"{base_loadpath}/evalDiffInit_pyramid_v2_stochastic_multiscale_ablation__d72_p80_train_1_eval_1/raw_prediction_forecastGT.pkl"
                },
            
            ############################################## trained DNNs tested on different S-Levels ##############################################
            "d-convlstm-slevel-100":{
                80: f"{base_loadpath}/evalDiffInitSlevel80_pyramid_v2_stochastic_multiscale_ablation__d72_p100_train_1_eval_1/raw_prediction_forecastGT.pkl",
                85: f"{base_loadpath}/evalDiffInitSlevel85_pyramid_v2_stochastic_multiscale_ablation__d72_p100_train_1_eval_1/raw_prediction_forecastGT.pkl",
                90: f"{base_loadpath}/evalDiffInitSlevel90_pyramid_v2_stochastic_multiscale_ablation__d72_p100_train_1_eval_1/raw_prediction_forecastGT.pkl",
                95: f"{base_loadpath}/evalDiffInitSlevel95_pyramid_v2_stochastic_multiscale_ablation__d72_p100_train_1_eval_1/raw_prediction_forecastGT.pkl",
                100: f"{base_loadpath}/evalDiffInitSlevel100_pyramid_v2_stochastic_multiscale_ablation__d72_p100_train_1_eval_1/raw_prediction_forecastGT.pkl",
            },       
            "d-convlstm-slevel-95":{
                80: f"{base_loadpath}/evalDiffInitSlevel80_pyramid_v2_stochastic_multiscale_ablation__d72_p95_train_1_eval_1/raw_prediction_forecastGT.pkl",
                85: f"{base_loadpath}/evalDiffInitSlevel85_pyramid_v2_stochastic_multiscale_ablation__d72_p95_train_1_eval_1/raw_prediction_forecastGT.pkl",
                90: f"{base_loadpath}/evalDiffInitSlevel90_pyramid_v2_stochastic_multiscale_ablation__d72_p95_train_1_eval_1/raw_prediction_forecastGT.pkl",
                95: f"{base_loadpath}/evalDiffInitSlevel95_pyramid_v2_stochastic_multiscale_ablation__d72_p95_train_1_eval_1/raw_prediction_forecastGT.pkl",
                100: f"{base_loadpath}/evalDiffInitSlevel100_pyramid_v2_stochastic_multiscale_ablation__d72_p95_train_1_eval_1/raw_prediction_forecastGT.pkl",
            },
            "d-convlstm-slevel-90":{
                80: f"{base_loadpath}/evalDiffInitSlevel80_pyramid_v2_stochastic_multiscale_ablation__d72_p90_train_1_eval_1/raw_prediction_forecastGT.pkl",
                85: f"{base_loadpath}/evalDiffInitSlevel85_pyramid_v2_stochastic_multiscale_ablation__d72_p90_train_1_eval_1/raw_prediction_forecastGT.pkl",
                90: f"{base_loadpath}/evalDiffInitSlevel90_pyramid_v2_stochastic_multiscale_ablation__d72_p90_train_1_eval_1/raw_prediction_forecastGT.pkl",
                95: f"{base_loadpath}/evalDiffInitSlevel95_pyramid_v2_stochastic_multiscale_ablation__d72_p90_train_1_eval_1/raw_prediction_forecastGT.pkl",
                100: f"{base_loadpath}/evalDiffInitSlevel100_pyramid_v2_stochastic_multiscale_ablation__d72_p90_train_1_eval_1/raw_prediction_forecastGT.pkl",
            },
            "d-convlstm-slevel-85":{
                80: f"{base_loadpath}/evalDiffInitSlevel80_pyramid_v2_stochastic_multiscale_ablation__d72_p85_train_1_eval_1/raw_prediction_forecastGT.pkl",
                85: f"{base_loadpath}/evalDiffInitSlevel85_pyramid_v2_stochastic_multiscale_ablation__d72_p85_train_1_eval_1/raw_prediction_forecastGT.pkl",
                90: f"{base_loadpath}/evalDiffInitSlevel90_pyramid_v2_stochastic_multiscale_ablation__d72_p85_train_1_eval_1/raw_prediction_forecastGT.pkl",
                95: f"{base_loadpath}/evalDiffInitSlevel95_pyramid_v2_stochastic_multiscale_ablation__d72_p85_train_1_eval_1/raw_prediction_forecastGT.pkl",
                100: f"{base_loadpath}/evalDiffInitSlevel100_pyramid_v2_stochastic_multiscale_ablation__d72_p85_train_1_eval_1/raw_prediction_forecastGT.pkl",
            },
            "d-convlstm-slevel-80":{
                80: f"{base_loadpath}/evalDiffInitSlevel80_pyramid_v2_stochastic_multiscale_ablation__d72_p80_train_1_eval_1/raw_prediction_forecastGT.pkl",
                85: f"{base_loadpath}/evalDiffInitSlevel85_pyramid_v2_stochastic_multiscale_ablation__d72_p80_train_1_eval_1/raw_prediction_forecastGT.pkl",
                90: f"{base_loadpath}/evalDiffInitSlevel90_pyramid_v2_stochastic_multiscale_ablation__d72_p80_train_1_eval_1/raw_prediction_forecastGT.pkl",
                95: f"{base_loadpath}/evalDiffInitSlevel95_pyramid_v2_stochastic_multiscale_ablation__d72_p80_train_1_eval_1/raw_prediction_forecastGT.pkl",
                100: f"{base_loadpath}/evalDiffInitSlevel100_pyramid_v2_stochastic_multiscale_ablation__d72_p80_train_1_eval_1/raw_prediction_forecastGT.pkl",
            },
            ###################################################################################################################
        },
    }
    
    path = raw_pred_forecastGT_dict[dataset_type][dnn_name][stochasticity_level]
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data




def plot_statisticGT_vs_dnnForecast(df, savepath_plots, bins=10):    
    plt.style.use('classic')

    # Filter out slevel of 0
    df = df[df['s-level'] != 0]

    plt.figure(figsize=(8, 5))
    s_levels = df['s-level'].unique()
    print(f"Unique s-levels: {s_levels}")
    
    sweep_list_og = [1,0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50]
    num_unique_colors = len(set(sweep_list_og))
    color_points = np.concatenate((np.linspace(0, 0.3, (num_unique_colors // 2)), np.linspace(0.7, 1, (num_unique_colors // 2)+1)))
    colors = plt.get_cmap('RdGy')(color_points)
    colors = [colors[i] for i,x in enumerate(sweep_list_og) if int(100-x*100) in s_levels]
    colors = colors[::-1]
    print(colors)

    # Define a list of marker types
    markers = ['o', 's', '^', 'v', '<']
    markers = markers[::-1]

    for i, s_level in enumerate(sorted(s_levels, reverse=True)):
        df_subset = df[df['s-level'] == s_level]
        df_subset['pred_bin'] = pd.cut(df_subset['pred'], bins=bins)
        df_summary = df_subset.groupby('pred_bin')['forecast-gt'].agg(['mean', 'std']).reset_index()
        df_summary['bin_center'] = df_summary['pred_bin'].apply(lambda x: x.mid)

        # Use different markers for each s-level
        marker = markers[i % len(markers)]
        
        plt.plot(df_summary['bin_center'], df_summary['mean'], linestyle='--', color=colors[i], linewidth=2, alpha=1, zorder=2, marker=marker, markersize=8, label=f'{s_level}')
        # # Add error bars
        # plt.errorbar(df_summary['bin_center'], df_summary['mean'], yerr=df_summary['std'], color=colors[i])
        
    plt.title('Trend of Forecast-GT with Prediction Bins', fontsize=16, fontweight='bold')
    plt.xlabel('Mean DNN Forecast', fontsize=20, fontweight='bold')
    plt.ylabel('Mean Statistic-GT', fontsize=20, fontweight='bold')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.plot([0, 1], [0, 1], 'k--', zorder=0, alpha=0.5)
    plt.xticks(np.arange(0, 1+1/bins, 1/bins), fontsize=12, fontweight='bold')
    
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')
    
    plt.legend(loc='best', title='S-Level')
    handles, labels = plt.gca().get_legend_handles_labels()
    # Here we reverse the order of handles and labels
    legend = plt.legend(handles[::-1], labels[::-1], title='S-Level', title_fontsize=22, fontsize=20, loc='upper left', frameon=True, edgecolor='black')
    plt.setp(legend.get_title(), fontweight='bold', fontsize=20)
    plt.setp(legend.get_texts(), fontweight='bold', fontsize=20)
    
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.1, zorder=0)
    plt.tight_layout()
    plot_filename = savepath_plots if savepath_plots else 'line_plot_pred_vs_forecastGT_markers.png'
    plt.savefig(plot_filename, dpi=300)
    plt.close()

def plot_histogram(df, savepath_plots, bins=10, inset=False):
    plt.style.use('classic')


    plt.figure(figsize=(12, 6))
    plt.grid(False)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3, zorder=0)
    
    s_levels = df['s-level'].unique()
    
    sweep_list_og = [1,0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50]
    num_unique_colors = len(set(sweep_list_og))
    color_points = np.concatenate((np.linspace(0, 0.3, (num_unique_colors // 2)), np.linspace(0.7, 1, (num_unique_colors // 2)+1)))
    colors = plt.get_cmap('RdGy')(color_points)
    colors = [colors[i] for i,x in enumerate(sweep_list_og) if int(100-x*100) in s_levels]
    # colors = colors[::-1]
    
    # Calculate the width of each bar, with some space between groups
    bar_width = (df['pred'].max() - df['pred'].min()) / (bins * (len(s_levels) + 1))

    # Determine the offset for each s-level
    offset = np.linspace(-bar_width*(len(s_levels)-1)/2, bar_width*(len(s_levels)-1)/2, len(s_levels))
    print(offset)
    
    hatch_list = [None, '\\\\', '..', '--', '//', 'x', 'o', 'O', '.', '*']

    for i, s_level in enumerate(sorted(s_levels, reverse=False)):  # Reverse the order to match the previous plot
        df_subset = df[df['s-level'] == s_level].copy()
        df_subset.loc[:, 'pred_bin'] = pd.cut(df_subset['pred'], bins=bins, include_lowest=True)
        bin_counts = df_subset.groupby('pred_bin').size()
        normalized_bin_counts = bin_counts / bin_counts.sum()
        
        # Calculate bin centers and adjust for offset
        bin_centers = [interval.mid for interval in df_subset['pred_bin'].cat.categories]
        bin_centers_adjusted = bin_centers + offset[i]
        print(bin_centers_adjusted)

        if inset:
            # Remove the first and last bin centers and counts 
            bin_centers_adjusted = bin_centers_adjusted[1:-1]
            normalized_bin_counts = normalized_bin_counts[1:-1]
        
        plt.bar(bin_centers_adjusted, normalized_bin_counts, 
                width=bar_width,
                color=colors[i], alpha=0.9, label=f'{s_level}', zorder=1, hatch=hatch_list[i],
                edgecolor='black', linewidth=1.5)

    plt.xlabel('Mean Prediction in Bin', fontsize=16, fontweight='bold')
    plt.ylabel('Normalized Bin Count', fontsize=16, fontweight='bold')

    # Match x-ticks rotation and style
    if not inset:
        plt.xticks(np.arange(0, 1 + bar_width, 1/bins), fontsize=16, fontweight='bold')
        plt.yticks(fontsize=16, fontweight='bold')

    else:
        plt.xticks(np.arange(0.1, 0.9 + bar_width, 1/bins), fontsize=18, fontweight='bold')
        plt.yticks(fontsize=18, fontweight='bold')
        

    if not inset:
        plt.legend(loc='best', title='S-Level')
        # Here we reverse the order of handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()
        # legend = plt.legend(handles[::-1], labels[::-1], title='S-Level', title_fontsize='16', fontsize=14, frameon=True, edgecolor='black')
        
        legend = plt.legend(handles, labels, title='S-Level', title_fontsize='16', fontsize=14, frameon=True, edgecolor='black')
        plt.setp(legend.get_title(), fontweight='bold', fontsize=14)
        plt.setp(legend.get_texts(), fontweight='bold', fontsize=14)

    plt.tight_layout()
    plot_filename = savepath_plots if savepath_plots else 'grouped_histogram_pred_vs_forecastGT.png'
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    
def consolidate_raw_pred_forecastGT_dict_SD(dnn_name, stochasticity_list, base_savepath, forecast_GT_folder):
    consolidated_dataframes = []
    
    for stochasticity_level in stochasticity_list:
        # Load the raw prediction and forecastGT data
        raw_pred_forecastGT_dict = load_raw_pred_forecastGT_dict(dnn_name, stochasticity_level,
                                                                 dataset_type="sameInit",
                                                                base_loadpath=forecast_GT_folder)
        # Load the SD vs time step dictionary
        SD_vs_time_step_dict = load_SD_vs_time_step_dict()
        
        for sample_idx, forecastGT_prediction in raw_pred_forecastGT_dict.items():
            # consolidated_dataframes = []
            
            print(f"Processing sample: {sample_idx}")
            prediction = forecastGT_prediction['prediction'] # (T,H,W)
            forecastGT = forecastGT_prediction['forecastGT'] # (T,H,W)
            
            for t in range(prediction.shape[0]):
                # Get the SD value for the given time step
                SD = SD_vs_time_step_dict[stochasticity_level][t+1] # Skip the first time step as no prediction is made for it
                
                # Get the prediction and forecastGT for the given time step
                pred = prediction[t]
                gt = forecastGT[t]
                
                # Flatten the prediction and forecastGT and store them in a dataframe along with the SD value
                pred_flat = pred.flatten()
                gt_flat = gt.flatten()
                
                for i in range(len(pred_flat)):
                    consolidated_dataframes.append([100-stochasticity_level, SD, t, pred_flat[i], gt_flat[i]])
            
            if sample_idx > 100:
                break
                
    # Create a dataframe from the consolidated dataframes
    df = pd.DataFrame(consolidated_dataframes, columns=['s-level', 'SD', 't', 'pred', 'forecast-gt'])
    plot_statisticGT_vs_dnnForecast(df, f"{base_savepath}/reliability_analysis/SD_pred_forecastGT.pdf")
    plot_histogram(df, f"{base_savepath}/reliability_analysis/histogram_pred_forecastGT.pdf", inset=False)
    plot_histogram(df, f"{base_savepath}/reliability_analysis/reliability_analysis/histogram_pred_forecastGT_inset.pdf", inset=True)
        
    return df
            

def main():     
    # Get the location of the folder where the raw predictions and forecastGT are stored
    script_folder = os.path.dirname(os.path.realpath(__file__))
    project_folder = os.path.dirname(os.path.dirname(script_folder))
    forecast_GT_folder = os.path.join(project_folder, "convLSTM_inference_runs")
    
    # Generate the dataframe for the given DNN and stochasticity level
    dnn_name = "d-convlstm"
    stochasticity_list = [100, 95, 90, 85, 80]
   
    # Create the base save path
    base_savepath = "Enter the base path where you want to save the plots"
    os.makedirs(base_savepath, exist_ok=True)
    
    # Generate the dataframe
    df = consolidate_raw_pred_forecastGT_dict_SD(dnn_name, 
                                            stochasticity_list,
                                            base_savepath,
                                            forecast_GT_folder)
    

if __name__ == "__main__":
    main()


