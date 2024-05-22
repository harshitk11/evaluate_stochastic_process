"""
Code used to generate the matrix of heatmaps showing the performance of the DNNs trained on different stochasticity levels
(Figure 5 and Figure 13 of the paper)
"""

from ece_counterexample_study import metric_vs_time_combined_simulation
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_average_metric(slevel_metric_timestep):
    average_slevel_metric = {}
    
    for slevel, metric_timestep in slevel_metric_timestep.items():
        average_slevel_metric[slevel] = {}
        
        for metric_name, metric_values_dict in metric_timestep.items():
            # get list of metric values for each timestep
            metric_values = metric_values_dict.values()
            # average the metric values
            avg_metric = sum(metric_values) / len(metric_values)
            # store the average metric value
            average_slevel_metric[slevel][metric_name] = avg_metric
            
    return average_slevel_metric

def create_heatmaps_from_metrics(dnn_average_metric, savepath=None, dnn_name_base="d-convlstm"):
    """
    Structure of the dnn_average_metric dictionary:
    {
        'dnn_name_1': {
            'slevel_1': {
                'metric_1': value_1,
                'metric_2': value_2,
                ...
            },
            'slevel_2': {
                'metric_1': value_1,
                'metric_2': value_2,
                ...
            },
            ...
        },
        'dnn_name_2': {
            'slevel_1': {
                'metric_1': value_1,
                'metric_2': value_2,
                ...
            },
            'slevel_2': {
                'metric_1': value_1,
                'metric_2': value_2,
                ...
            },
            ...
        },
        ...
    }
    """
    # Identify all unique metric names across all DNNs and slevels
    all_metric_names = set()
    for dnn_metrics in dnn_average_metric.values():
        for slevel_metrics in dnn_metrics.values():
            all_metric_names.update(slevel_metrics.keys())
    
    # Process each metric individually to create separate heatmaps
    for metric_name in all_metric_names:
        # Initialize a list to hold data for the current metric
        data = []
        index_labels = []  # To keep track of row labels (DNN names)
        column_labels = [100,95,90,85,80]  # To keep track of all slevels encountered
        dnn_name_list = [f"{dnn_name_base}-slevel-100",
                         f"{dnn_name_base}-slevel-95",
                         f"{dnn_name_base}-slevel-90",
                         f"{dnn_name_base}-slevel-85",
                         f"{dnn_name_base}-slevel-80"]
        
        # invert the dnn_name_list
        dnn_name_list = dnn_name_list[::-1]
        
        # Extract data for the current metric from the nested dictionary
        for dnn_name in dnn_name_list:
            slevel_dict = dnn_average_metric[dnn_name]
            row = []
            index_labels.append(int(dnn_name.split('-')[-1]))
            
            for slevel in column_labels:
                if slevel in slevel_dict:
                    value = slevel_dict[slevel].get(metric_name)
                    if metric_name in ["auc_pr","recall"]:
                        value = 1 - value 
                    row.append(value)
                else:
                    row.append(float('nan'))        
            data.append(row)
        
        # Substract 100 from the column and row labels
        index_labels = [100 - label for label in index_labels]
        column_labels = [100 - label for label in column_labels]
        
        # Create DataFrame with the collected data
        df = pd.DataFrame(data, index=index_labels, columns=column_labels)
        
        plt.figure(figsize=(12, 6))
        
        heatmap = sns.heatmap(df, annot=True, fmt=".3f", cmap='RdPu', linewidths=.5, annot_kws={'size': 22, 'weight': 'bold'})
        
        # Create all caps title for the heatmap
        if metric_name == "auc_pr":
            metric_name_title = "1 - AUC-PR"
        elif metric_name == "recall":
            metric_name_title = "1 - RECALL"
        else:   
            metric_name_title = metric_name.upper()
            
        heatmap.set_title(f'{metric_name_title}', fontdict={'fontsize':22, 'weight': 'bold'}, pad=10)
        heatmap.set_ylabel('DNN trained on S-level', fontsize=24)
        heatmap.set_xlabel('DNN tested on S-level', fontsize=24)
        plt.xticks(rotation=45, fontsize=22)
        plt.yticks(rotation=0, fontsize=22)
        plt.tight_layout()
        # increase the size of the text in colorbar 
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=18)
        
        # reduce space between colorbar and heatmap
        plt.subplots_adjust(right=0.7)
        
        
        if savepath:
            fpath = os.path.join(savepath, dnn_name_base, f'{metric_name}_heatmap.pdf')
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            plt.savefig(fpath, format='pdf', bbox_inches='tight')
        
        
# get the slevel_metric_timestep dict for each dnn_name_trained_slevel
def main():
    dnn_name_base = "d-convlstm" # or "arnca","d-convlstm","convlstmbtk","multilayer_convlstm"
    dnn_name_list = [f"{dnn_name_base}-slevel-100", 
                     f"{dnn_name_base}-slevel-95", 
                     f"{dnn_name_base}-slevel-90", 
                     f"{dnn_name_base}-slevel-85", 
                     f"{dnn_name_base}-slevel-80"]
    stochasticity_list = [100, 95, 90, 85, 80]
    
    base_savepath = "Enter the base path to save the plots"
    os.makedirs(base_savepath, exist_ok=True)
    
    dnn_average_metric = {}
    for dnn_name in dnn_name_list:
        print(f"*********************** Processing {dnn_name} ***********************")
        slevel_metric_timestep = metric_vs_time_combined_simulation.generate_metrics_combined_simulation(stochasticity_list, 
                                                                                                         dnn_name, 
                                                                                                         base_savepath, 
                                                                                                         dataset_type= "diffInit", 
                                                                                                         num_samples=300,
                                                                                                         load_pickle=True)
        average_slevel_metric = calculate_average_metric(slevel_metric_timestep)
        dnn_average_metric[dnn_name] = average_slevel_metric
    
    
    # pretty print the dnn_average_metric dictionary
    for dnn_name, slevel_metrics in dnn_average_metric.items():
        print(f"{dnn_name}:")
        for slevel, metric_values in slevel_metrics.items():
            print(f"  S-level {slevel}:")
            for metric_name, metric_value in metric_values.items():
                print(f"    {metric_name}: {metric_value}")
                
    create_heatmaps_from_metrics(dnn_average_metric, savepath=base_savepath, dnn_name_base=dnn_name_base)
    
if __name__ == "__main__":
    main()