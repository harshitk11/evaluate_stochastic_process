import json
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import csv

def enhance_plot(ax):
    # Add grid
    ax.grid(linestyle='--', linewidth=0.7, alpha=0.7)

    # # Set the background color
    # ax.set_facecolor('whitesmoke')

    # Set the grid style
    ax.grid(linestyle='--', linewidth=0.7, alpha=0.7)

    # Customize the tick labels
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Set the font
    font = {'weight': 'bold', 'size': 14}
    plt.rc('font', **font)

    return ax

def compute_confidence_interval(mean, std_dev, n, confidence_level=0.95, proportion=False):
    if proportion:
        # Wilson Score Interval for binomial proportions
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        denominator = 1 + z**2 / n
        centre_adjusted_probability = mean + z**2 / (2 * n)
        adjusted_standard_deviation = np.sqrt((mean * (1 - mean) + z**2 / (4 * n)) / n)
        
        lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
        upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    else:
        std_err = std_dev / (n ** 0.5)
        degree_of_freedom = n - 1  
        critical_value = stats.t.ppf((1 + confidence_level) / 2, degree_of_freedom)
        margin_of_error = critical_value * std_err
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
        
    return lower_bound, upper_bound

def compare_two_metrics(json_path, start_time, time_gap, n_list, metric1, metric2, save_path=None, num_iterations=30):
    experiment_name = json_path.split('/')[-2]
    save_path = os.path.join(save_path, experiment_name)
    os.makedirs(save_path, exist_ok=True)
        
    # Load the data from JSON file
    with open(json_path, 'r') as json_file:
        raw_metric_data = json.load(json_file)

    metrics = [metric1, metric2]
    
    # Set subplot with 2 rows and 1 column
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 14))
    
    for ax, metric_name in zip(axes, metrics):
        ax = enhance_plot(ax)
        metric_data = raw_metric_data[metric_name]

        # Convert keys to integers and filter by start_time and time_gap
        times = sorted([int(t) for t in metric_data['1'].keys() if int(t) >= start_time])[::time_gap]
        
        # Set bar width
        width = 0.15

        # Set position for each bar
        r = np.arange(len(times))

        for i, n in enumerate(n_list):
            conf_intervals_lower = []
            conf_intervals_upper = []
            conf_intervals_std_dev = []

            for t in times:
                str_t = str(t)
                if str_t in metric_data['1']:
                    lower_aggregate = []
                    upper_aggregate = []
                    
                    # Ensure all values are within the [0, 1] range
                    valid_scores = [score for score in metric_data['1'][str_t] if 0 <= score <= 1]
                    invalid_scores = [score for score in metric_data['1'][str_t] if score < 0 or score > 1]
                    if len(valid_scores) != len(metric_data['1'][str_t]):
                        print(f"Warning [{metric_name}]: Invalid values found for time {t} - {invalid_scores}.")
                        
                    
                    # Iterate for num_iterations and aggregate the confidence intervals
                    for _ in range(num_iterations):
                        n = min(n, len(metric_data['1'][str_t]))
                        eval_scores = np.random.choice(metric_data['1'][str_t], n, replace=False)
                        mean = np.mean(eval_scores)
                        std_dev = np.std(eval_scores, ddof=1)
                        lower, upper = compute_confidence_interval(mean, std_dev, n)
                        lower_aggregate.append(lower)
                        upper_aggregate.append(upper)

                    # Average the confidence intervals and calculate the standard deviation
                    avg_lower = np.mean(lower_aggregate)
                    avg_upper = np.mean(upper_aggregate)
                    std_dev_of_intervals = np.std(upper_aggregate)  # Using upper bounds, can also use lower or both
                    conf_intervals_lower.append(avg_lower)
                    conf_intervals_upper.append(avg_upper)
                    conf_intervals_std_dev.append(std_dev_of_intervals)
                else:
                    conf_intervals_lower.append(np.nan)
                    conf_intervals_upper.append(np.nan)
                    conf_intervals_std_dev.append(np.nan)
            
            bar_heights = [upper - lower for lower, upper in zip(conf_intervals_lower, conf_intervals_upper)]
            bar_position = [r[j] + i * width for j in range(len(r))]
            
            # Adding error bars to represent variance in confidence intervals
            ax.bar(bar_position, bar_heights, width=width, label=f'{n}', yerr=conf_intervals_std_dev, capsize=5)
            
        ax.legend(loc='upper left', fontsize=12, title="Sample Size (n)", title_fontsize='14', labelspacing=0.5)
        ax.set_xticks(r + width/2 * (len(n_list)-1))
        ax.set_xticklabels([str(t) for t in times])
        # Set axis labels, title, and legend
        legend = ax.legend(loc='upper left', fontsize=12, labelspacing=0.5)
        legend.set_title("Sample Size (n)", prop={'size': 14, 'weight': 'bold'})
        ax.set_xlabel('Time Step', fontsize=14, fontweight='bold')
        ax.set_ylabel('Confidence Interval Width', fontsize=14, fontweight='bold')
        if metric_name == 'ROC_PR':
            mtitle = 'AUC_PR'
        else:
            mtitle = metric_name
        ax.set_title(f'Confidence Intervals for {mtitle}', fontsize=16, fontweight='bold')
        

    # Adjust y-limit for both subplots to be the same
    max_ylim = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    # max_ylim = 1
    axes[0].set_ylim(top=max_ylim)
    axes[1].set_ylim(top=max_ylim)
    axes[0].grid(linestyle='--')
    axes[1].grid(linestyle='--')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'{metric1}_vs_{metric2}_confidence_intervals.png'))


def plot_confidence_intervals(json_path, start_time, time_gap, n_list, save_path=None, num_iterations=30):
    experiment_name = json_path.split('/')[-2]
    save_path = os.path.join(save_path, experiment_name)
    os.makedirs(save_path, exist_ok=True)
        
    with open(json_path, 'r') as json_file:
        raw_metric_data = json.load(json_file)

    for metric_name, metric_data in raw_metric_data.items():
        times = sorted([int(t) for t in metric_data['1'].keys() if int(t) >= start_time])[::time_gap]
        width = 0.15
        fig, ax = plt.subplots(figsize=(15, 7))
        ax = enhance_plot(ax)
        r = np.arange(len(times))

        for i, n in enumerate(n_list):
            conf_intervals_lower = []
            conf_intervals_upper = []
            conf_intervals_std_dev = []

            for t in times:
                str_t = str(t)
                if str_t in metric_data['1']:
                    lower_aggregate = []
                    upper_aggregate = []

                    for _ in range(num_iterations):
                        n = min(n, len(metric_data['1'][str_t]))
                        eval_scores = np.random.choice(metric_data['1'][str_t], n, replace=False)
                        mean = np.mean(eval_scores)
                        std_dev = np.std(eval_scores, ddof=1)
                        lower, upper = compute_confidence_interval(mean, std_dev, n)
                        lower_aggregate.append(lower)
                        upper_aggregate.append(upper)

                    avg_lower = np.mean(lower_aggregate)
                    avg_upper = np.mean(upper_aggregate)
                    std_dev_of_intervals = np.std(upper_aggregate)
                    conf_intervals_lower.append(avg_lower)
                    conf_intervals_upper.append(avg_upper)
                    conf_intervals_std_dev.append(std_dev_of_intervals)
                else:
                    conf_intervals_lower.append(np.nan)
                    conf_intervals_upper.append(np.nan)
                    conf_intervals_std_dev.append(np.nan)

            bar_heights = [upper - lower for lower, upper in zip(conf_intervals_lower, conf_intervals_upper)]
            bar_position = [r[j] + i * width for j in range(len(r))]
            ax.bar(bar_position, bar_heights, width=width, label=f'n={n}', yerr=conf_intervals_std_dev, capsize=5)

        ax.set_xticks(r + width/2 * (len(n_list)-1))
        ax.set_xticklabels([str(t) for t in times])
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Confidence Interval Width')
        ax.set_title(f'Confidence Intervals for {metric_name}')
        ax.legend()
        ax.grid()

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, f'{metric_name}_confidence_intervals.png'))
        plt.close()


def plot_confidence_interval_width_vs_stochastic_n(metric_name, stochastic_list, n_list, start_time, time_gap, num_iterations, save_path=None):
    plt.figure(figsize=(8, 8))
    
    # Choose a color map that provides distinct colors and is visually pleasing in print
    colors = sns.color_palette("rocket", len(stochastic_list))
    line_styles = ['-', '--', '-.', ':'] * (len(n_list) // 4 + 1)
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'H', '+', 'x', '|', '_'] * (len(n_list) // 14 + 1)

    for idx, stochastic in enumerate(stochastic_list):
        color = colors[idx]
        marker = markers[idx]
        json_file_path = f'/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/evaluator/evalSameInit_pyramid_v2_stochastic_multiscale_ablation__d72_p{stochastic}_train_1_eval_1/raw_metric_data.json'
        with open(json_file_path, 'r') as json_file:
            metric_data = json.load(json_file)[metric_name]

        times = sorted([int(t) for t in metric_data['1'].keys() if int(t) >= start_time])[::time_gap]
        
        for i, n in enumerate(n_list):
            line_style = line_styles[i]
        
            conf_intervals_lower = []
            conf_intervals_upper = []
            conf_intervals_std_dev = []

            for t in times:
                str_t = str(t)
                if str_t in metric_data['1']:
                    lower_aggregate = []
                    upper_aggregate = []

                    for _ in range(num_iterations):
                        n = min(n, len(metric_data['1'][str_t]))
                        eval_scores = np.random.choice(metric_data['1'][str_t], n, replace=False)
                        mean = np.mean(eval_scores)
                        std_dev = np.std(eval_scores, ddof=1)
                        lower, upper = compute_confidence_interval(mean, std_dev, n)
                        lower_aggregate.append(lower)
                        upper_aggregate.append(upper)

                    avg_lower = np.mean(lower_aggregate)
                    avg_upper = np.mean(upper_aggregate)
                    std_dev_of_intervals = np.std(upper_aggregate)
                    conf_intervals_lower.append(avg_lower)
                    conf_intervals_upper.append(avg_upper)
                    conf_intervals_std_dev.append(std_dev_of_intervals)
                else:
                    conf_intervals_lower.append(np.nan)
                    conf_intervals_upper.append(np.nan)
                    conf_intervals_std_dev.append(np.nan)

            interval_widths = [upper - lower for lower, upper in zip(conf_intervals_lower, conf_intervals_upper)]
            plt.plot(times, interval_widths, label=f"Stochastic={stochastic}, n={n}", linestyle=line_style, color=color, marker=marker, markersize=6, linewidth=2)

    plt.ylim(0.0, 0.3)
    plt.xlabel('Time Step', fontsize=14, fontweight='bold')
    plt.ylabel(f'Confidence Interval Width of {metric_name}', fontsize=14, fontweight='bold')
    # plt.title(f'Confidence Interval Width of {metric_name} for Different Stochastic and n values', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    
    plt.grid(linestyle='--', linewidth=0.7, alpha=0.7)  # Added grid with alpha
    plt.tight_layout()

    if save_path:
        print(f"Saving plot to {save_path}")
        save_path = os.path.join(save_path, f'{metric_name}_confidence_interval_width_vs_stochastic_n.png')
        plt.savefig(save_path)

    plt.close()

def plot_avg_confidence_interval_width_vs_n(metrics, stochastic_list, n_list, start_time, time_gap, num_iterations, save_path=None, inset=False):
    plt.figure(figsize=(10, 8))
    
    # Color and style setup
    colors = sns.color_palette("tab10", len(metrics))  # Providing distinct colors for different metrics
    line_styles = ['-', '--', '-.', ':'] * (len(stochastic_list) // 4 + 1)  # Line styles for different stochastic values
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'H', '+', 'x', '|', '_'] * (len(metrics) // 14 + 1)

    for metric_idx, metric_name in enumerate(metrics):
        color = colors[metric_idx]
        marker = markers[metric_idx]

        for idx, stochastic in enumerate(stochastic_list):
            line_style = line_styles[idx]
            
            json_file_path = f'/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/evaluator/evalSameInit_pyramid_v2_stochastic_multiscale_ablation__d72_p{stochastic}_train_1_eval_1/raw_metric_data.json'
            with open(json_file_path, 'r') as json_file:
                metric_data = json.load(json_file)[metric_name]

            times = sorted([int(t) for t in metric_data['1'].keys() if int(t) >= start_time])[::time_gap]
            
            avg_widths = []
            for n in n_list:
                aggregated_widths = []
                for t in times:
                    str_t = str(t)
                    if str_t in metric_data['1']:
                        lower_aggregate = []
                        upper_aggregate = []

                        for _ in range(num_iterations):
                            n = min(n, len(metric_data['1'][str_t]))
                            eval_scores = np.random.choice(metric_data['1'][str_t], n, replace=False)
                            mean = np.mean(eval_scores)
                            std_dev = np.std(eval_scores, ddof=1)
                            lower, upper = compute_confidence_interval(mean, std_dev, n)
                            lower_aggregate.append(lower)
                            upper_aggregate.append(upper)
                        
                        interval_widths = [upper - lower for lower, upper in zip(lower_aggregate, upper_aggregate)]
                        aggregated_widths.extend(interval_widths)
                
                # Compute average width for the current n
                avg_width = sum(aggregated_widths) / len(aggregated_widths)
                # avg_width = -np.log10(avg_width)  # Convert to log scale
                avg_widths.append(avg_width)
            
            # Plot for the current metric and stochastic value
            plt.plot(n_list, avg_widths, label=f"{metric_name}, S-Level={stochastic}", linestyle=line_style, color=color, linewidth=2.5, marker=marker, markersize=8, markeredgecolor='k', markeredgewidth=0.5)

    if not inset:
        plt.xlabel('Sample Size (n)', fontsize=16, fontweight='bold')
        plt.ylabel('Average Confidence Interval Width', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=14, fontweight='bold')  # Adjust font size and weight for tick labels on x-axis
    plt.yticks(fontsize=14, fontweight='bold')  # Adjust font size and weight for tick labels on y-axis
    if not inset:
        plt.legend(fontsize=14, title_fontsize='16')
        # plt.legend(fontsize=14, title="Metrics", title_fontsize='16')
    plt.grid(linestyle='--', linewidth=0.7, alpha=0.7)  # Adding grid with transparency
    plt.gca().invert_xaxis()  
    plt.tight_layout()

    if save_path:
        metrics_str = "_".join(metrics)
        if inset:
            save_path = os.path.join(save_path, f'avg_confidence_interval_width_vs_n_{metrics_str}_inset.pdf')
        else:
            save_path = os.path.join(save_path, f'avg_confidence_interval_width_vs_n_{metrics_str}.pdf')
        print(f"Saving plot to {save_path}")
        plt.savefig(save_path)

    plt.close() 

class Heatmap:
    @staticmethod
    def bootstrap_confidence_interval(data, statistic, num_iterations, n, alpha=0.05):
        """
        Bootstraps the data to compute the confidence interval of a given statistic.
        """
        bootstrap_samples = [np.random.choice(data, size=n, replace=True) for _ in range(num_iterations)]
        values = [statistic(sample) for sample in bootstrap_samples]
        lower_bound = np.percentile(values, 100*(alpha/2))
        upper_bound = np.percentile(values, 100*(1 - alpha/2))
        return lower_bound, upper_bound

    
    @staticmethod
    def compute_heatmap_data(json_file_path, metric_name, start_time, time_gap, num_iterations, n_list):
        with open(json_file_path, 'r') as json_file:
            metric_data = json.load(json_file)[metric_name]

        times = sorted([int(t) for t in metric_data['1'].keys() if int(t) >= start_time])[::time_gap]
        
        heatmap_data = []
        for n in n_list:
            interval_widths_row = []
            for t in times:
                str_t = str(t)
                if str_t in metric_data['1']:
                    # Using bootstrapping to compute confidence intervals directly from the data
                    lower, upper = Heatmap.bootstrap_confidence_interval(metric_data['1'][str_t], np.mean, num_iterations, n)
               
                    if upper - lower > 1:
                        print(f"Warning [{metric_name}]: Interval width greater than 1 for time {t} and n {n}. CI: ({lower}, {upper})")
                        # continue
                
                    interval_width = upper - lower
                    interval_widths_row.append(interval_width)
                else:
                    interval_widths_row.append(np.nan)
            
            heatmap_data.append(interval_widths_row)
        
        return np.array(heatmap_data)
    
        
    @staticmethod
    def save_to_csv(metric_name, stochastic_list, n_list, start_time, time_gap, num_iterations, csv_file_path, experiment_prefix = "evalSameInit"):
        data_matrix = []
        
        for stochastic in stochastic_list:
            json_file_path = f'/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/evaluator/{experiment_prefix}_pyramid_v2_stochastic_multiscale_ablation__d72_p{stochastic}_train_1_eval_1/raw_metric_data.json'
            data_matrix.append(Heatmap.compute_heatmap_data(json_file_path, metric_name, start_time, time_gap, num_iterations, n_list))
        
        data_matrix = np.array(data_matrix)
        avg_data = np.mean(data_matrix, axis=2)  # average over time steps
        base_value = avg_data[0, 0]
        percentage_increase = abs(avg_data - base_value) / base_value
        
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Writing the header if the file is empty
            if file.tell() == 0:
                header = ['Metric', 'S-Level'] + n_list
                writer.writerow(header)
            
            # Writing the rows
            for i, stochastic in enumerate(stochastic_list):
                stochastic = 100 - stochastic # S-level
                row = [metric_name, f"{stochastic}%"] + [f"{avg_data[i, j]:.3f} ({percentage_increase[i, j]:.0f}x)" for j in range(len(n_list))]
                writer.writerow(row)

    
def main():
    # for stochastic in [95, 90, 85, 80]:
    #     json_file_path = f'/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/evaluator/evalSameInit_pyramid_v2_stochastic_multiscale_ablation__d72_p{stochastic}_train_1_eval_1/raw_metric_data.json'  # Replace with your actual path
    #     save_path = '/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/confidence_intervals'  
    #     os.makedirs(save_path, exist_ok=True)
        
    #     start_time = 10  
    #     time_gap = 5  
    #     n_list = [1000, 500, 200, 50, 10, 5]  

    #     # plot_confidence_intervals(json_file_path, start_time, time_gap, n_list,save_path=save_path, num_iterations=600)
    #     # compare_list = [("F1", "Optimal_F1_PR"), ("F1", "MSE"), ("F1", "ROC_AUC"), ("F1", "ROC_PR"), ("F1", "ROC_AUC")]
    #     compare_list = [("F1", "MAE")]
    #     for metric1,metric2 in compare_list:
    #         print(f"Stochastic: {stochastic}, Metric1: {metric1}, Metric2: {metric2}")
    #         compare_two_metrics(json_file_path, start_time, time_gap, n_list, 
    #                             metric1=metric1, 
    #                             metric2=metric2, 
    #                             save_path=save_path,
    #                             num_iterations=500)
    #         # exit()
            
    # save_path = '/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/confidence_intervals/consolidated'
    # os.makedirs(save_path, exist_ok=True)
    # for metric_name in ['F1', 'Optimal_F1_PR', 'MSE', 'ROC_AUC', 'ROC_PR']:
    #     plot_confidence_interval_width_vs_stochastic_n(metric_name = metric_name, 
    #                                                 stochastic_list = [95,90,85,80], 
    #                                                 #    stochastic_list = [95,80], 
    #                                                 #    n_list=[50,10,5], 
    #                                                 n_list=[50, 5], 
    #                                                 start_time=10, 
    #                                                 time_gap=5, 
    #                                                 num_iterations=100, 
    #                                                 save_path=save_path)
    
    # save_path = '/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/confidence_intervals/consolidated'
    # os.makedirs(save_path, exist_ok=True)
    # plot_avg_confidence_interval_width_vs_n(metrics = ['F1', 'MSE', 'ROC_AUC'],
    #                                         # stochastic_list = [95,90,85,80], 
    #                                         stochastic_list = [95,80], 
    #                                         n_list = [500, 300, 100, 50],
    #                                         # n_list = [500, 300, 100, 50, 20, 10, 5],
    #                                         start_time = 10, 
    #                                         time_gap = 5, 
    #                                         num_iterations = 10, 
    #                                         save_path=save_path,
    #                                         inset=True)
    
    
    # HEATMAP
    # save_path = '/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/confidence_intervals/heatmaps'
    # os.makedirs(save_path, exist_ok=True)
    # for metric_name in ['F1', 'Accuracy', 'MSE', 'MAE', 'BCE', 'ROC_AUC', 'ROC_PR']:
    #     for calculate_percentage in [True, False]:
    #         Heatmap.plot_heatmap(metric_name, 
    #                             stochastic_list=[95,90,85,80], 
    #                             n_list=[50, 20, 10, 5, 2], 
    #                             start_time=10, 
    #                             time_gap=5, 
    #                             num_iterations=500, 
    #                             save_path=save_path,
    #                             calculate_percentage=calculate_percentage)
            
    save_path = '/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/confidence_intervals/csv'
    os.makedirs(save_path, exist_ok=True)
    # for metric_name in ['F1', 'Accuracy', 'Precision', 'Recall', 'MSE', 'MAE', 'BCE', 'ROC_AUC', 'ROC_PR']:
    for metric_name in ['SSIM']:
        print(f"Metric: {metric_name}")
    #     # Heatmap.print_latex_table(metric_name, 
        #                         stochastic_list=[95,90,85,80], 
        #                         n_list=[50, 20, 10, 5, 2], 
        #                         start_time=10, 
        #                         time_gap=5, 
        #                         num_iterations=500)
        Heatmap.save_to_csv(metric_name, 
                                stochastic_list=[95,90,85,80], 
                                n_list=[50, 20, 10, 5, 2, 1], 
                                start_time=10, 
                                time_gap=5, 
                                num_iterations=1000,
                                csv_file_path=os.path.join(save_path, f'metric_bootstrapped.csv'),
                                experiment_prefix='backup_evalSameInit')
if __name__ == '__main__':
    main()