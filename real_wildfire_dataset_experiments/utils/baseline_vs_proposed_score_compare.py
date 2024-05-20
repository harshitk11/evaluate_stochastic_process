# Description: This script is used to compare the scores of the baseline and proposed models.
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import bootstrap
import pandas as pd
import seaborn as sns

def get_scores_from_json(file_path, scale, metric):
    """
    Reads a JSON file and returns the scores based on the provided scale and metric.
    
    Parameters:
    - file_path: str, path to the JSON file.
    - scale: scale for which scores are to be retrieved.
    - metric: str, the type of metric ("f1_scores", "roc_auc_scores", etc.)
    
    Returns:
    - numpy array of scores corresponding to the specified scale and metric.
    """
    # Reading the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        scores_dict = json.load(file)
    
    # Extracting the specified scores
    try:
        scores = scores_dict[metric][str(scale)]  # Convert scale to string as keys in JSON are strings
    except KeyError:
        raise KeyError(f"No scores found for metric '{metric}' and scale '{scale}' in the file '{file_path}'.")
    
    return scores

def get_top_x_percent_indices(file_path, scale, metric, x):
    """
    Retrieves the indices of the top x% scores from a JSON file along with the scores.
    
    Parameters:
    - file_path: str, path to the JSON file.
    - scale: scale for which scores are to be retrieved.
    - metric: str, the type of metric ("f1_scores", "roc_auc_scores", etc.)
    - x: float, percentage of top scores to retrieve.
    
    Returns:
    - list of tuples where each tuple contains an index and the corresponding score.
    """
    # Getting scores using the get_scores_from_json function
    scores = get_scores_from_json(file_path, scale, metric)
    
    # Getting the indices sorted by score
    sorted_indices = np.argsort(scores)[::-1]
    
    # Calculating the number of top scores to retrieve
    top_x_num = int(len(scores) * x / 100)
    
    # Getting the top x% indices and scores
    top_x_indices_scores = [(index, scores[index]) for index in sorted_indices[:top_x_num]]
    
    return top_x_indices_scores

def get_top_x_percent_diff_indices(baseline_path, proposed_path, scale, metric, x):
    """
    Retrieves the indices of the top x% of scores where the difference 
    between two sets of scores from two JSON files is the highest.
    
    Parameters:
    - file_path1: str, path to the first JSON file.
    - file_path2: str, path to the second JSON file.
    - scale: scale for which scores are to be retrieved.
    - metric: str, the type of metric ("f1_scores", "roc_auc_scores", etc.)
    - x: float, percentage of top differences to retrieve.
    
    Returns:
    - list of tuples where each tuple contains an index and the corresponding difference in scores.
    """
    # Getting scores from both JSON files
    scores_baseline = get_scores_from_json(baseline_path, scale, metric)
    scores_proposed = get_scores_from_json(proposed_path, scale, metric)
    
    if len(scores_baseline) != len(scores_proposed):
        raise ValueError("The number of scores in both files does not match.")
    
    # Calculating the differences
    differences = [(s1 - s2) for s1, s2 in zip(scores_proposed, scores_baseline)]
    
    # Getting the indices sorted by differences
    sorted_indices = np.argsort(differences)[::-1]
    
    # Calculating the number of top differences to retrieve
    top_x_num = int(len(differences) * x / 100)
    
    # Getting the top x% indices and differences
    top_x_indices_differences = [(index, differences[index]) for index in sorted_indices[:top_x_num]]
    
    return top_x_indices_differences

class MC_Score_Plotter_overall:
    @staticmethod
    def create_box_plot(model_config, num_MC_simulations, metric, scale):
        bottleneck_flag = model_config["bottleneck_flag"]
        bottleneck_channel = model_config["bottleneck_channel"]
        metric_score_MC_list = []
        
        for i in range(num_MC_simulations):
            file_path = f"/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/raw_evaluation_scores/eval_segmentation_model_300epochs_MC_run{i}__bottleneck{bottleneck_flag}_kernelSize3_bottleneckChannel{bottleneck_channel}/raw_scores.json"    
            scores = get_scores_from_json(file_path, scale, metric)
            mean_score = np.nanmean(scores)
            metric_score_MC_list.append(mean_score)
        
        # Print the best score and it's MC run
        print(f" ******* Model Config: {model_config} ******* ")
        best_score = np.nanmax(metric_score_MC_list)
        best_score_MC_run = metric_score_MC_list.index(best_score)
        worst_score = np.nanmin(metric_score_MC_list)
        worst_score_MC_run = metric_score_MC_list.index(worst_score)
        print(f"Best {metric} score: {best_score} at MC run {best_score_MC_run}")
        print(f"Worst {metric} score: {worst_score} at MC run {worst_score_MC_run}")
                
        return metric_score_MC_list
    
    @staticmethod
    def create_labels_from_config(model_config_list):
        """
        Generates labels for box plots based on model configurations.
        """
        labels = []
        for config in model_config_list:
            bottleneck_status = "Bottleneck" if config['bottleneck_flag'] else "No Bottleneck"
            channel_count = config['bottleneck_channel']
            labels.append(f'{bottleneck_status}, {channel_count}')
        return labels

    @staticmethod
    def plot_combined_box_plots(model_config_list, num_MC_simulations, metric, scale, save_path=None):
        """
        Plots combined box plots for all model configurations.
        """
        # Getting scores for all model configurations
        scores_list = []
        for model_config in model_config_list:
            scores = MC_Score_Plotter_overall.create_box_plot(model_config, num_MC_simulations, metric, scale)
            scores_list.append(scores)
        
        # Labels for x-axis based on model configuration
        labels = MC_Score_Plotter_overall.create_labels_from_config(model_config_list)
        
        # Creating a box plot
        plt.figure(figsize=(10, 6))
        plt.boxplot(scores_list, labels=labels)
        plt.title('Model Configuration vs. Metric Score')
        plt.xlabel('Model Configuration')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            print("Saving the box plot...")
            os.makedirs(save_path, exist_ok=True)
            fpath = os.path.join(save_path, f"{metric}_box_plot_overall.png")
            plt.savefig(fpath)
        plt.close()
        
    
class MC_Score_Plotter_stochasticity_stratified:
    @staticmethod
    def load_stochasticity_estimates(file_path):
        """
        Returns a dictionary containing the stochasticity estimates for each scale.
        format: {sample_idx: stochasticity_estimate} e.g., {
                                                                "0": {
                                                                    "Jaccard_Similarity": 0.0,
                                                                    "Dice_Similarity": 0.0
                                                                }, ...
        """
        # Reading the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            stochasticity_dict = json.load(file)

        return stochasticity_dict
    
    @staticmethod
    def create_score_stochasticity_dict(model_config, MC_run, metric, scale):
        """
        Returns a dictionary containing the scores and stochasticity estimates for each sample.
        Format: {sample_idx: {"score": score_val, "stochasticity": stochasticity_dict}} 
        """
        bottleneck_flag = model_config["bottleneck_flag"]
        bottleneck_channel = model_config["bottleneck_channel"]
        
        # Load the scores
        score_file_path = f"/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/raw_evaluation_scores/eval_segmentation_model_300epochs_MC_run{MC_run}__bottleneck{bottleneck_flag}_kernelSize3_bottleneckChannel{bottleneck_channel}/raw_scores.json"    
        scores_list = get_scores_from_json(score_file_path, scale, metric)
        
        print(f" ******* Model Config: {model_config} ******* ")
        print(f"Loaded scores for MC run {MC_run} with mean score: {np.nanmean(scores_list)}")
        
        # Load the stochasticity estimates
        stochasticity_file_path = f"/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/stochasticity_estimates/eval_segmentation_model_300epochs_MC_run{MC_run}__bottleneck{bottleneck_flag}_kernelSize3_bottleneckChannel{bottleneck_channel}/stochasticity_estimates.json"
        stochasticity_dict = MC_Score_Plotter_stochasticity_stratified.load_stochasticity_estimates(stochasticity_file_path)
        
        stochasticity_score_dict = {}
        for sample_idx, score_val in enumerate(scores_list):
            stochasticity_score_dict[sample_idx] = {"score": score_val, "stochasticity": stochasticity_dict[str(sample_idx)]}
        
        return stochasticity_score_dict
    
    @staticmethod
    def create_combinedMCruns_score_stochasticity_dict(model_config, num_MC_simulations, metric, scale, MC_index=None):
        """
        Returns a dictionary containing the scores for each sample across all MC runs and stochasticity estimates.
        Format: {sample_idx: {"score": [score_val1, score_val2, ...], "stochasticity": stochasticity_dict}, ...}
        """
        combined_dict = {}
        
        # Determine the range of MC runs to process based on whether MC_index is provided
        mc_runs_to_process = range(num_MC_simulations) if MC_index is None else [MC_index]
        
        for MC_run in mc_runs_to_process:
            stochasticity_score_dict = MC_Score_Plotter_stochasticity_stratified.create_score_stochasticity_dict(model_config, MC_run, metric, scale)
            for sample_idx, score_stochasticity in stochasticity_score_dict.items():
                if sample_idx in combined_dict:
                    combined_dict[sample_idx]["score"].append(score_stochasticity["score"])
                    
                    # Check if the stochasticity estimates are the same
                    if combined_dict[sample_idx]["stochasticity"] != score_stochasticity["stochasticity"]:
                        raise ValueError("The stochasticity estimates are not the same.")
                else:
                    combined_dict[sample_idx] = {"score": [score_stochasticity["score"]], "stochasticity": score_stochasticity["stochasticity"]}
                    
        return combined_dict

    
    @staticmethod
    def plot_stratified_performance(model_config_list, num_MC_simulations, metric, scale, save_path=None, best_MC_run=False):
        bar_width = 0.2
        patterns = ['/', '\\', None, '.']
        labels_list = MC_Score_Plotter_overall.create_labels_from_config(model_config_list)
        
        plt.figure(figsize=(12, 6))
        plt.style.use('ggplot')
        
        print(f"\nPlotting stratified performance for metric: {metric}")
        for config_index, model_config in enumerate(model_config_list):
            print(f"[{config_index}] Processing model config: {model_config}")
            MC_index = model_config["best_MC_run"] if best_MC_run else None
            combined_dict = MC_Score_Plotter_stochasticity_stratified.create_combinedMCruns_score_stochasticity_dict(model_config, num_MC_simulations, metric, scale,
                                                                                                                    MC_index=MC_index)
            
            binned_scores = {i: [] for i in range(10)}
            
            # Calculate the mean score for each sample using the scores from all MC runs and bin them based on the stochasticity estimates
            for sample_data in combined_dict.values():
                score_list = sample_data["score"]
                mean_score = np.nanmean(score_list) if not all(np.isnan(score_list)) else 0
                # mean_score = np.nanmax(score_list) if not all(np.isnan(score_list)) else 0
                stochasticity_value = sample_data["stochasticity"]["Dice_Similarity"]
                bin_index = min(int(stochasticity_value * 10), 9)
                binned_scores[bin_index].append(mean_score)
            
            # Calculate the mean and standard deviation of the scores for each bin
            x_positions, y_means, y_stds, supports = [], [], [], []
            for bin_index, scores in binned_scores.items():
                x_positions.append(bin_index + config_index * bar_width)
                y_means.append(np.mean(scores) if scores else 0)
                y_stds.append(np.std(scores) if scores else 0)  # Calculate standard deviation
                supports.append(len(scores))
            
            bars = plt.bar(x_positions, y_means, width=bar_width, label=f'Model Config {config_index + 1}', hatch=patterns[config_index], yerr=y_stds, capsize=2)  # Use yerr and add capsize for better visualization
            # bars = plt.bar(x_positions, y_means, width=bar_width, label=labels_list[config_index], hatch=patterns[config_index])  # Use yerr and add capsize for better visualization
            
            if config_index == 2:
                ytot = [ym+yerr for ym,yerr in zip(y_means, y_stds)]
                # ytot = y_means
                for xpos, ypos, support in zip(x_positions, ytot, supports):
                    if metric == "mse_scores":
                        plt.text(xpos, ypos + 0.001, f'n={support}', ha='center', va='bottom')
                    else:
                        plt.text(xpos, ypos + 0.01, f'n={support}', ha='center', va='bottom')
            
        plt.xlabel('Stochasticity Bins', fontsize=14)
        plt.ylabel(f'Mean {metric}', fontsize=14)
        plt.title(f'Stratified Performance by Binned Stochasticity ({metric})', fontsize=16)
        plt.xticks([i + bar_width * (len(model_config_list) - 1) / 2 for i in range(10)],
                [f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(10)], fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fpath = os.path.join(save_path, f"{metric}_stratified_performance.png")
            plt.savefig(fpath)
        plt.close()

class confidence_interval_generator:
    @staticmethod
    def calculate_confidence_interval(data, num_bootstrap_samples=1000, confidence_level=0.95):
        # filter out nan values
        data = data[~np.isnan(data)]
        
        if data.shape[0] < 2:
            return np.nan, np.nan
        
        # Define a function to compute the mean of the sample
        def mean_func(sample, axis):
            return np.mean(sample, axis=axis)

        # Perform bootstrap
        res = bootstrap((data,), mean_func, n_resamples=num_bootstrap_samples, method='percentile')
        lower_bound = res.confidence_interval.low
        upper_bound = res.confidence_interval.high
        return lower_bound, upper_bound
    
    @staticmethod
    def create_table_from_model_configs(model_config_list, num_MC_simulations, metric, scale, num_bootstrap_samples=1000, best_MC_run=False, fpath=None,):
        records = []
        
        print(f"\nPlotting stratified performance for metric: {metric}")
        for config_index, model_config in enumerate(model_config_list):
            print(f"[{config_index}] Processing model config: {model_config}")
            MC_index = model_config["best_MC_run"] if best_MC_run else None
            combined_dict = MC_Score_Plotter_stochasticity_stratified.create_combinedMCruns_score_stochasticity_dict(model_config, num_MC_simulations, metric, scale,
                                                                                                                    MC_index=MC_index)
            
            all_scores = []
            binned_scores = {i: [] for i in range(10)}
            for sample_data in combined_dict.values():
                score_list = sample_data["score"]
                all_scores.append(score_list[0])
                
                mean_score = np.nanmean(score_list) if not all(np.isnan(score_list)) else 0
                stochasticity_value = sample_data["stochasticity"]["Dice_Similarity"]
                bin_index = min(int(stochasticity_value * 10), 9)
                binned_scores[bin_index].append(mean_score)
            
            for bin_index, scores in binned_scores.items():
                if scores:
                    mean_score = np.mean(scores)
                    support = len(scores)
                    lower_ci, upper_ci = confidence_interval_generator.calculate_confidence_interval(np.array(scores), num_bootstrap_samples)
                    record = {
                        'Model_Config': f'{model_config["bottleneck_flag"]}-{model_config["bottleneck_channel"]}',
                        'support': support,
                        'Dice_Similarity_Bin': f'{bin_index/10:.1f}-{(bin_index+1)/10:.1f}',
                        'Mean_Metric_Score': mean_score,
                        'Lower_CI': lower_ci,
                        'Upper_CI': upper_ci
                    }
                    records.append(record)
            
            # Print the overall confidence interval
            overall_lower_ci, overall_upper_ci = confidence_interval_generator.calculate_confidence_interval(np.array(all_scores), num_bootstrap_samples)
            print(f"Overall CI: {overall_lower_ci}, {overall_upper_ci}")

            if fpath is not None:
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                sns.set(style="whitegrid", palette="muted", font_scale=1.5)
                plt.figure(figsize=(8, 6))
                sns.histplot(all_scores, bins=20, kde=False, color='gray', edgecolor='black', linewidth=0.8, hatch='//')

                # Customizing ticks and labels to be bold
                plt.xticks(fontweight='bold')
                plt.yticks(fontweight='bold')

                if metric == "auc_pr_scores":
                    plt.xlabel("AUC-PR Score", fontweight='bold')
                else:
                    plt.xlabel(f'{metric}', fontweight='bold')
                plt.ylabel('Frequency', fontweight='bold')
                
                # If you have a title, uncomment the line below and set your title
                # plt.title(f'Model Config: {model_config}, Metric: {metric}', fontweight='bold')

                plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
                plt.tight_layout()
                plt.savefig(fpath, dpi=300)
                plt.close()
        records = pd.DataFrame(records)
        
        print(records)
        
        
        return records
        
def main():
    # ################################################# Inspecting the scores #################################################
    # # Loading scores from JSON files
    # file_path_proposed = "/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/raw_evaluation_scores/eval_segmentation_model_300epochs_MC_run28__bottleneckFalse_kernelSize3_bottleneckChannel16/raw_scores.json"
    # file_path_baseline = "/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/raw_evaluation_scores/eval_segmentation_model_300epochs_MC_run10__bottleneckTrue_kernelSize3_bottleneckChannel16/raw_scores.json"
    # scale = "1"
    # metric = "auc_pr_scores" 
    # # metric = "mse_scores" 
    # scores = get_scores_from_json(file_path_baseline, scale, metric)
    # # print(scores, len(scores))  
    # print(np.nanmax(scores), np.nanmin(scores), np.nanmean(scores), np.nanstd(scores))
    
    # # # Getting the indices of the top x% scores
    # # x = 10 
    # # indices = get_top_x_percent_indices(file_path_proposed, scale, metric, x)
    # # print(indices)
    
    # # # Getting the indices of the top x% differences
    # # indices_differences = get_top_x_percent_diff_indices(file_path_proposed, file_path_baseline, scale, metric, x)
    # # print(indices_differences)
    # ########################################################################################################################
    
    # ################################################# Plotting the overall performance #################################################
    # model_config = {"bottleneck_flag": False, "bottleneck_channel": 16}
    # num_MC_simulations = 30
    # metric = "bce_scores"
    # # metric = "auc_pr_scores"
    # # metric ="auc_pr_score_all"
    # # metric = "recall_scores"
    # # metric = "precision_scores"
    # scale = "1"
    # model_config_list = [{"bottleneck_flag": True, "bottleneck_channel": 16},
    #                  {"bottleneck_flag": True, "bottleneck_channel": 32},
    #                  {"bottleneck_flag": False, "bottleneck_channel": 16},
    #                  {"bottleneck_flag": False, "bottleneck_channel": 32}]
    
    # # print(MC_Score_Plotter_overall.create_box_plot(model_config, num_MC_simulations, metric, scale))
    
    # savepath = "/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/performance_visualizations/overall"
    # MC_Score_Plotter_overall.plot_combined_box_plots(model_config_list, num_MC_simulations, metric, scale, save_path=savepath)
    # exit()
    # ###################################################################################################################################
    
    # ################################################# Plotting the stratified performance #################################################
    # num_MC_simulations = 30
    # for metric in ["auc_pr_scores", "recall_scores", "precision_scores","mse_scores", "f1_scores"]:
    #     scale = "1"
    #     model_config_list = [{"bottleneck_flag": True, "bottleneck_channel": 16, "best_MC_run": 10, "worst_MC_run": 26},
    #                     {"bottleneck_flag": True, "bottleneck_channel": 32, "best_MC_run": 29, "worst_MC_run": 16},
    #                     {"bottleneck_flag": False, "bottleneck_channel": 16, "best_MC_run": 28, "worst_MC_run": 13},
    #                     {"bottleneck_flag": False, "bottleneck_channel": 32, "best_MC_run": 13, "worst_MC_run": 10}]
        
    #     # model_config_list = [{"bottleneck_flag": True, "bottleneck_channel": 16, "best_MC_run": 26, "worst_MC_run": 26},
    #     #                 {"bottleneck_flag": True, "bottleneck_channel": 32, "best_MC_run": 16, "worst_MC_run": 16},
    #     #                 {"bottleneck_flag": False, "bottleneck_channel": 16, "best_MC_run": 28, "worst_MC_run": 13},
    #     #                 {"bottleneck_flag": False, "bottleneck_channel": 32, "best_MC_run": 13, "worst_MC_run": 10}]
    #     savepath = "/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/performance_visualizations/stochasticity_stratified"
    #     MC_Score_Plotter_stochasticity_stratified.plot_stratified_performance(model_config_list, num_MC_simulations, metric, scale, 
    #                                                                           save_path=savepath,
    #                                                                         #   save_path=None,
    #                                                                           best_MC_run=True)
    # #######################################################################################################################################
    
    ################################################# Plotting the CI table #################################################
    num_MC_simulations = 30
    for metric in ["auc_pr_scores", "recall_scores", "precision_scores","mse_scores"]:
    # for metric in ["auc_pr_scores"]:
        print(f"\nProcessing metric: {metric}")
        scale = "1"
        model_config_list = [
                        {"bottleneck_flag": True, "bottleneck_channel": 16, "best_MC_run": 10, "worst_MC_run": 26},
                        # {"bottleneck_flag": True, "bottleneck_channel": 32, "best_MC_run": 29, "worst_MC_run": 16},
                        # {"bottleneck_flag": False, "bottleneck_channel": 16, "best_MC_run": 28, "worst_MC_run": 13},
                        # {"bottleneck_flag": False, "bottleneck_channel": 32, "best_MC_run": 13, "worst_MC_run": 10},
                        ]
        
        # model_config_list = [{"bottleneck_flag": True, "bottleneck_channel": 16, "best_MC_run": 26, "worst_MC_run": 26},
        #                 {"bottleneck_flag": True, "bottleneck_channel": 32, "best_MC_run": 16, "worst_MC_run": 16},
        #                 {"bottleneck_flag": False, "bottleneck_channel": 16, "best_MC_run": 28, "worst_MC_run": 13},
        #                 {"bottleneck_flag": False, "bottleneck_channel": 32, "best_MC_run": 13, "worst_MC_run": 10}]
        savepath = "/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/performance_visualizations/metric_histograms"
        confidence_interval_generator.create_table_from_model_configs(model_config_list, num_MC_simulations, metric, scale, num_bootstrap_samples=1000, best_MC_run=True,
                                                                      fpath=os.path.join(savepath, f"{metric}_histogram.pdf"))
    
    #######################################################################################################################################
    

if __name__ == '__main__':
    main()