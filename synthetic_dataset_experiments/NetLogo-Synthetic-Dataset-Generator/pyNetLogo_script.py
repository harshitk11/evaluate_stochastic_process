"""
Used for creating Figure 4 in the paper
"""
import pyNetLogo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import numpy as np
import pickle
import hashlib
import matplotlib.gridspec as gridspec


sns.set_style('white')
sns.set_context('talk')

class visualization_tools:
    """
    Contains all the methods to visualize the output extracted from netlogo.
    """
    @staticmethod
    def plot_netlogo_map(patchAttr_df, save_path=None):
        """
        Plots the netlogo map using the dataframe patchAttr_df.
        """
        fig, ax = plt.subplots(1)
        patches = sns.heatmap(patchAttr_df.clip(lower=0, axis=1), xticklabels=False, yticklabels=False, ax=ax, cbar=False)
        ax.tick_params(left=False, bottom=False)
        ax.set_aspect('equal')
        fig.set_size_inches(8,8)

        if save_path:
            plt.savefig(save_path)
        plt.close('all')

    @staticmethod
    def generate_gif(folder_path, gif_savePath):
        # Get a list of all the PNG files in the folder
        file_names = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        file_names = sorted(file_names, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        
        # Create a list to store the Image objects
        images = []

        # Loop through the file names and open each file as an Image object
        for file_name in file_names:
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)
            # Add the Image object to the list of images
            images.append(image)

        # Save the list of images as a GIF animation
        if gif_savePath:
            images[0].save(gif_savePath, save_all=True, append_images=images[1:], optimize=False, duration=100, loop=1)


            
class netlogo_wrapper:
    """
    Contains all the methods to interact with a netlogo simulation.
    """ 
    @staticmethod
    def setup_netlogo_environment(**kwargs):
        """
        Sets up the NetLogo environement using pyNetLogo
        params:
            - kwargs: Contains all the parameters required to setup the netlogo environment
                    {
                        "netlogo_binary_path": Location where Netlogo is installed,
                        "netlogo_model_path": Location of the Netlogo model to run,
                        ... Model specific parameters ... 
                    }
        Output:
            - netlogo_run : Instance of the netlogo link that is used for the simulation
        """
        netlogo_run = pyNetLogo.NetLogoLink(gui=False, 
                                        thd=False, 
                                        netlogo_home=kwargs["netlogo_binary_path"],
                                        netlogo_version='6.0')

        netlogo_run.load_model(kwargs["netlogo_model_path"])

        # Set the parameters
        netlogo_run.command(f'set density {kwargs["density"]}')
        netlogo_run.command(f'set seed_intensity {kwargs["seed_intensity"]}')
        netlogo_run.command(f'set transfer_efficiency {kwargs["transfer_efficiency"]}')
        netlogo_run.command(f'set q_th {kwargs["q_th"]}')
        netlogo_run.command(f'set q_die {kwargs["q_die"]}')
        netlogo_run.command(f'set fire_radius {kwargs["fire_radius"]}')
        netlogo_run.command(f'set random_fire_spawn {kwargs["random_fire_spawn"]}')
        netlogo_run.command(f'set num_seeds {kwargs["num_seeds"]}')
        netlogo_run.command(f'set show_sensors {kwargs["show_sensors"]}')
        netlogo_run.command(f'set record_video {kwargs["record_video"]}')
        netlogo_run.command(f'set stochastic_meter {kwargs["stochastic_meter"]}')
        netlogo_run.command(f'set run_number {kwargs["run_number"]}')
        netlogo_run.command(f'set save_path "{kwargs["save_path"]}"')
        
        # Setup the run
        netlogo_run.command('setup')
        return netlogo_run

    @staticmethod
    def run_netlogoSimulation_for_N_ticks(netlogoRun, N):
        """
        Moves forward N time ticks in the netlogoRun.
        """
        netlogoRun.repeat_command(netlogo_command='go', reps=N)
        return netlogoRun

    @staticmethod
    def read_patch_attribute(netlogoRun, patchAttribute):
        """
        Reads the patchAttribute from the netlogoRun and returns a dataframe.
        """
        patchAttr_df = netlogoRun.patch_report(patchAttribute)
        return patchAttr_df
    
    @staticmethod
    def write_patch_attribute(netlogoRun, patchAttribute, patchAttr_df):
        """
        Updates the patchAttribute in the netlogoRun with the values contained in dataframe (patchAttr_df)
        """
        netlogoRun.patch_set(patchAttribute, patchAttr_df)

        
    @staticmethod
    def killNetlogo(netlogoRun):
        netlogoRun.kill_workspace()

        
    
class DensityOfStates:
    
    @staticmethod
    def maxpooling_2d(df, kernel_size):
        '''Perform max-pooling on 2D dataframe'''
        pooled_data = []
        for i in range(0, df.shape[0], kernel_size):
            row = []
            for j in range(0, df.shape[1], kernel_size):
                row.append(df.iloc[i:i+kernel_size, j:j+kernel_size].max().max())
            pooled_data.append(row)
        return pd.DataFrame(pooled_data)
    
    def generate_temporal_DOS_distribution(num_MC_runs=1000, density=80, stochastic_meter=0.50, save_path=None, heterogenous_agents=False,  scales=[1,2,4,8,16], debug=False):
        """
        Used to generate a distribution of the number of unburnt trees at every timestep of the simulation
        params:
            - num_MC_runs: Number of monte carlo runs to perform
            - density: Density of the forest
            - stochastic_meter: Stochasticity of the forest
            - save_path: Path to save the histogram
            - heterogenous_agents: Boolean to indicate if the agents are heterogenous or not
        Output:
            - num_unburnt_trees_dict: Dictionary containing the number of unburnt trees at the end of the simulation
        """
        num_time_steps = 300
        # Setup the netlogo model
        kwargs = {"netlogo_binary_path": "/home/hkumar64/netlogo/NetLogo6",
                  "netlogo_model_path":"./netlogo_models/forest_fire_evolution_stochastic_heterogenous_64grid.nlogo",
                  "density": density,
                  "seed_intensity": 3,
                  "transfer_efficiency": 1,
                  "q_th": 5,
                  "q_die": 3,
                  "fire_radius": 1,
                  "random_fire_spawn": True,
                  "num_seeds": 3,
                  "show_sensors": False,
                  "record_video": False,
                  "stochastic_meter": stochastic_meter,
                  "run_number":1,
                  "save_path": None}
        
        # Dictionary to store the number of unburnt trees at the end of the simulation
        num_unburnt_trees_dict = {} 
        
        # For N monte carlo simulations
        for i in range(num_MC_runs):
            print(f"Run number: {i}")    
            # Perform the simulation
            netlogo_run = netlogo_wrapper.setup_netlogo_environment(**kwargs)    
            num_unburnt_trees_dict[i] = {}
            
            for t in range(num_time_steps):
                num_unburnt_trees_dict[i][t] = {}
                netlogo_run = netlogo_wrapper.run_netlogoSimulation_for_N_ticks(netlogoRun=netlogo_run, N=1)
                
                # Read the number of unburnt trees at the end of the simulation
                patchAttr_df = netlogo_wrapper.read_patch_attribute(netlogoRun = netlogo_run, patchAttribute ='pcolor')
                total_unburnt_trees_at_t = 0  # Total unburnt trees at current time t across all scales

                for scale in scales:
                    scaled_patchAttr_df = DensityOfStates.maxpooling_2d(patchAttr_df, scale) if scale > 1 else patchAttr_df
                    num_unburnt_trees = np.count_nonzero(scaled_patchAttr_df.values == 55)
                    num_unburnt_trees_dict[i][t][scale] = num_unburnt_trees
                    total_unburnt_trees_at_t += num_unburnt_trees

            netlogo_wrapper.killNetlogo(netlogoRun=netlogo_run)

            
        if save_path:
            fig, axs = plt.subplots(len(scales), 1, figsize=(8, 6 * len(scales)))
            for ax, scale in zip(axs, scales):
                all_values = np.array([v[scale] for v in num_unburnt_trees_dict.values()])
                mean = np.mean(all_values)
                std = np.std(all_values)

                iqr = np.percentile(all_values, 75) - np.percentile(all_values, 25)
                if iqr == 0:
                    if np.std(all_values) == 0:  # All values are identical
                        num_bins = 1
                        bins = [all_values.min(), all_values.max() + 1]  # +1 to ensure correct binning
                    else:
                        bin_width = np.std(all_values) / 10  # Just an arbitrary choice for default bin width
                        num_bins = int(np.ceil((all_values.max() - all_values.min()) / bin_width))
                        bins = np.linspace(all_values.min(), all_values.max(), num_bins)
                else:
                    bin_width = 2 * iqr / (len(all_values) ** (1/3))
                    num_bins = int(np.ceil((all_values.max() - all_values.min()) / bin_width))
                    bins = np.linspace(all_values.min(), all_values.max(), num_bins)

                ax.hist(all_values, bins=bins, color='steelblue', edgecolor='white')
                ax.set_xlabel("Number of unburnt trees", fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)
                ax.set_title(f"Scale: {scale}, Mean: {mean:.2f}, Std: {std:.2f}", fontsize=14)
                ax.tick_params(labelsize=10)
                ax.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        
        return num_unburnt_trees_dict
    
    
    
    def stochasticity_sweep_temporal_DOS(num_MC_runs=1000, stochasticity_list=[1], density=80, scales=[1,2,4,8,16], pickle_path=None):
        """
        Generates a dictionary of the number of unburnt trees at the end of the simulation (for each MC simulation) for each stochasticity
        params:
            - num_MC_runs: Number of monte carlo runs to perform
            - stochasticity_list: List of stochasticity values to sweep over
            - density: Density of the forest
        Return:
            - stochasticity_vs_num_unburnt_trees_dict: Dictionary containing the number of unburnt trees at the end of the simulation
                                                Format: {stochasticity: {MC_run_number: {scale: num_unburnt_trees}}}
        """
        # Load from checkpoint if it exists
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as handle:
                print(f"Loading from checkpoint: {pickle_path}")
                stochasticity_vs_num_unburnt_trees_dict = pickle.load(handle)
        else:
            stochasticity_vs_num_unburnt_trees_dict = {}
            
        for stochastic_meter in stochasticity_list:
            print(f"Stochasticity: {stochastic_meter}")
            if stochastic_meter in stochasticity_vs_num_unburnt_trees_dict.keys():
                print(f" - Skipping stochasticity: {stochastic_meter}")
                continue
            
            num_unburnt_trees_dict = DensityOfStates.generate_temporal_DOS_distribution(num_MC_runs=num_MC_runs, density=density, stochastic_meter=stochastic_meter, scales=scales)
            stochasticity_vs_num_unburnt_trees_dict[stochastic_meter] = num_unburnt_trees_dict
            
            # # Print Summary statistics for each scale
            # for scale in scales:
            #     values_at_scale = [val_dict[scale] for val_dict in num_unburnt_trees_dict.values()]
            #     print(f" [Summary] -> Density: {density}, Scale: {scale},  Average number of unburnt trees:{np.mean(values_at_scale)}\n")
            
            # save the dictionary
            with open(pickle_path, 'wb') as handle:
                pickle.dump(stochasticity_vs_num_unburnt_trees_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    
    @staticmethod
    def plot_temporal_density_stochasticity_sweep_dos(loadpath, save_path, scales, sweep_list, max_time_step=None):
        with open(loadpath, 'rb') as handle:
            sweepVal_vs_num_unburnt_trees_dict = pickle.load(handle)

        num_unique_colors = len(set(sweep_list))
        colors = plt.get_cmap('icefire')(np.linspace(0, 1, num_unique_colors))

        for scale in scales:
            plt.figure(figsize=(15, 10))
            plt.style.use('seaborn-whitegrid')  # Use a nicer style

            for color_idx, sweep_val in enumerate(sweep_list):
                num_unburnt_trees_dict = sweepVal_vs_num_unburnt_trees_dict[sweep_val]

                time_series_data = []
                
                if max_time_step is None:
                    time_step_list = sorted(num_unburnt_trees_dict[0].keys())
                else:
                    time_step_list = range(max_time_step)
                    
                for time_step in sorted(time_step_list):
                    values_at_time = np.array([run_dict[time_step][scale] for run_dict in num_unburnt_trees_dict.values() if time_step in run_dict.keys()])
                    mean_value = np.mean(values_at_time)
                    var_value = np.std(values_at_time)
                    time_series_data.append((mean_value, var_value))
                
                mean_values, var_values = zip(*time_series_data)
                time_steps = np.arange(len(mean_values))
                plt.plot(time_steps, mean_values, color=colors[color_idx], label=f"Stochasticity: {round(1-sweep_val, 2)}", linewidth=2)  # Increased linewidth
                plt.fill_between(time_steps, np.subtract(mean_values, var_values), np.add(mean_values, var_values), color=colors[color_idx], alpha=0.2)  # Increased alpha for visibility

            plt.xlabel("Time", fontsize=18, fontweight='bold', labelpad=10)  # Increased fontsize and added padding
            plt.ylabel("Number of Unburnt Trees", fontsize=18, fontweight='bold', labelpad=10)  # Increased fontsize and added padding
            plt.title(f"Temporal Density of Stochasticity Sweep at Scale {scale}", fontsize=20, fontweight='bold', pad=20)  # Added title with padding
            plt.xticks(fontsize=14)  # Increased tick fontsize
            plt.yticks(fontsize=14)  # Increased tick fontsize
            plt.legend(fontsize=12, loc='upper right', frameon=True, framealpha=0.9, title='Legend')  # Styled the legend
            plt.tight_layout()
            plt.savefig(f"{save_path}_scale_{scale}.pdf", dpi=300, bbox_inches='tight')  # Added bounding box tight
            plt.close()

    @staticmethod
    def plot_std_dev_over_time(loadpath, save_path, scales, sweep_list, max_time_step=None):
        with open(loadpath, 'rb') as handle:
            sweepVal_vs_num_unburnt_trees_dict = pickle.load(handle)

        num_unique_colors = len(set(sweep_list))
        color_points = np.concatenate((np.linspace(0, 0.3, (num_unique_colors // 2)), np.linspace(0.7, 1, (num_unique_colors // 2)+1)))
        colors = plt.get_cmap('RdGy')(color_points)
        
        marker_styles = ['o', 's', '^', 'v', '<', '>', 'p', 'X', '+', 'D', '|', '_']

        for scale in scales:
            plt.figure(figsize=(12, 10))
            plt.style.use('seaborn-whitegrid')

            for color_idx, sweep_val in enumerate(sweep_list):
                num_unburnt_trees_dict = sweepVal_vs_num_unburnt_trees_dict[sweep_val]
                time_series_data = []
                
                if max_time_step is None:
                    time_step_list = sorted(num_unburnt_trees_dict[0].keys())
                else:
                    time_step_list = range(max_time_step)
                    
                for time_step in sorted(time_step_list):
                    values_at_time = np.array([run_dict[time_step][scale] for run_dict in num_unburnt_trees_dict.values() if time_step in run_dict.keys()])
                    var_value = np.std(values_at_time)
                    time_series_data.append(var_value)
                
                time_steps = np.arange(len(time_series_data))
                marker_style = marker_styles[color_idx % len(marker_styles)]
                plt.plot(time_steps, time_series_data, color=colors[color_idx], label=f"{1-sweep_val:.2f}", linewidth=2, marker=marker_style, markersize=4, markerfacecolor='white', markeredgewidth=1.5, markeredgecolor=colors[color_idx])
            
            plt.xlabel("Time", fontsize=20, fontweight='bold', labelpad=12)
            plt.ylabel("Standard Deviation", fontsize=20, fontweight='bold', labelpad=12)
            
            for label in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
                label.set_fontweight('bold')
                label.set_fontsize(16)
            
            handles, labels = plt.gca().get_legend_handles_labels()
            # Here we reverse the order of handles and labels
            legend = plt.legend(handles[::-1], labels[::-1], title='S-Level', title_fontsize='16', fontsize=14, loc='center right', frameon=True, edgecolor='black')
            plt.setp(legend.get_title(), fontweight='bold')
            plt.setp(legend.get_texts(), fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(f"{save_path}_std_dev_scale_{scale}.pdf", dpi=300, bbox_inches='tight')
            plt.close()

            
    
    @staticmethod
    def plot_composite_visualization(loadpath, save_path, scales, sweep_list, max_time_step=None, x_upper_bound=20000):
        marker_styles = ['o', 's', '^', 'v', '<', '>', 'p', 'X', '+', 'D', 'P', '_', ]
        hatch_styles = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']  # A list of hatch styles

        with open(loadpath, 'rb') as handle:
            sweepVal_vs_num_unburnt_trees_dict = pickle.load(handle)

        num_unique_colors = len(set(sweep_list))
        color_points = np.concatenate((np.linspace(0, 0.3, (num_unique_colors // 2)), np.linspace(0.7, 1, (num_unique_colors // 2)+1)))
        colors = plt.get_cmap('RdGy')(color_points)
        
        for scale in scales:
            fig = plt.figure(figsize=(12, 8))
            plt.style.use('seaborn-whitegrid')

            ax1 = plt.subplot(111)  # Single subplot
            final_data_for_histogram = {}

            for color_idx, sweep_val in enumerate(sweep_list):
                num_unburnt_trees_dict = sweepVal_vs_num_unburnt_trees_dict[sweep_val]
                time_series_data = []
                values_at_final_time = []

                if max_time_step is None:
                    time_step_list = sorted(num_unburnt_trees_dict[0].keys())
                else:
                    time_step_list = range(max_time_step)

                for time_step in sorted(time_step_list):
                    values_at_time = np.array([run_dict[time_step][scale] for run_dict in num_unburnt_trees_dict.values() if time_step in run_dict.keys()])
                    mean_value = np.mean(values_at_time)
                    var_value = np.std(values_at_time)
                    time_series_data.append((mean_value, var_value))
                    if time_step == time_step_list[-1]:
                        values_at_final_time = values_at_time

                mean_values, var_values = zip(*time_series_data)
                time_steps = np.arange(len(mean_values))
                marker_style = marker_styles[color_idx % len(marker_styles)]
                hatch_style = hatch_styles[color_idx % len(hatch_styles)]  

                percentage_label = round((1-sweep_val) * 100)
                ax1.plot(time_steps, mean_values, color=colors[color_idx], label=percentage_label, linewidth=2, marker=marker_style, markersize=9, markerfacecolor='white', markeredgewidth=1.5, markeredgecolor=colors[color_idx], markevery=5)
                ax1.fill_between(time_steps, np.subtract(mean_values, var_values), np.add(mean_values, var_values), color=colors[color_idx], alpha=0.5, edgecolor=colors[color_idx], hatch=hatch_style)  # Applying hatch styles here

                final_data_for_histogram[sweep_val] = values_at_final_time  
            
            ax1.set_xlabel("Time", fontsize=20, fontweight='bold', labelpad=12)
            ax1.set_ylabel("Number of Unburnt Trees", fontsize=20, fontweight='bold', labelpad=12)
            ax1.grid(True, linestyle='--', alpha=0.5)

            # Increase the size of xticks and yticks and make them bold
            plt.xticks(fontsize=24, fontweight='bold')  # Increase tick font size and make bold
            plt.yticks(fontsize=24, fontweight='bold')  # Increase tick font size and make bold

            # Add legend to ax1
            handles, labels = ax1.get_legend_handles_labels()
            legend = ax1.legend(handles[::-1], labels[::-1], title='S-Level', title_fontsize='14', fontsize=14, edgecolor='black', frameon=True)
            plt.setp(legend.get_title(), fontweight='bold', fontsize='20')
            plt.setp(legend.get_texts(), fontweight='bold', fontsize='20')

            fig.tight_layout()
            plt.savefig(f"{save_path}_composite_scale_{scale}.pdf", dpi=300, bbox_inches='tight')
            plt.close()
            
    

class dataset_generation: 
    @staticmethod
    def generate_uniform_distribution_dataset(num_runs, density, stochastic_meter, base_path_rootdirectory): 
        """
        Generates num_runs of uniform distribution dataset for the given density.
        params:
            - num_runs: Number of runs to perform
            - density: Density of the forest
            - base_path_rootdirectory: Path where the png files will be saved
        Ouput:
            - Saves the png files in the base_path_rootdirectory
        """
        assert isinstance(stochastic_meter, float) and len(str(stochastic_meter).split(".")) == 2, "stochastic_meter should be a float with a number after the decimal point"
        stoch_prob = str(stochastic_meter).split(".")[-1]

        # Path where the npy files will be saved
        save_path = os.path.join(base_path_rootdirectory, f"uniform_{density}_p_{stoch_prob}", "raw_png")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        for i in range(num_runs):
            # Unique has to track each run
            params_str = f"density_{density}_run_{i}_num_runs_{num_runs}"
            unique = hashlib.md5(str(params_str).encode()).hexdigest()         
            experiment_save_path = os.path.join(save_path, f"experiment_{unique}")
            print(f"Run number: {i} | Hash: {unique}")
            
            # Setup the netlogo model
            kwargs = {"netlogo_binary_path": "/home/hkumar64/netlogo/NetLogo6",
                    "netlogo_model_path":"./netlogo_models/forest_fire_evolution_stochastic_heterogenous_64grid.nlogo",
                    "density": density,
                    "seed_intensity": 3,
                    "transfer_efficiency": 1,
                    "q_th": 5,
                    "q_die": 3,
                    "fire_radius": 1,
                    "random_fire_spawn": True,
                    "num_seeds": 3,
                    "show_sensors": False,
                    "record_video": True,
                    "stochastic_meter": stochastic_meter,
                    "run_number":i,
                    "save_path": experiment_save_path}
        
            # Perform the simulation
            netlogo_run = netlogo_wrapper.setup_netlogo_environment(**kwargs)
            netlogo_run = netlogo_wrapper.run_netlogoSimulation_for_N_ticks(netlogoRun=netlogo_run, N=1000)
            netlogo_wrapper.killNetlogo(netlogoRun=netlogo_run)
    
    
    @staticmethod
    def generate_stochastic_dataset():
        """
        Used for generating the synthetic dataset
        """
        stochastic_meter_list = [1.00,0.95,0.90,0.85,0.80]
        for stochastic_meter in stochastic_meter_list:
            dataset_generation.generate_uniform_distribution_dataset(num_runs=1000, 
                                                                    density=72,
                                                                    stochastic_meter=stochastic_meter, 
                                                                    base_path_rootdirectory="/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_same_init_1")

def main():
    ################# Used for generating Figure 3 in the paper #################
    # Temporal evolution of the standard deviation
    pickle_path = './res/temporal_300_stochasticity_vs_num_unburnt_trees_dict_64grid_same_fire_12345_same_forest_47822.pickle'
    DensityOfStates.stochasticity_sweep_temporal_DOS(num_MC_runs=1000, 
                                            stochasticity_list=[1,0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50], 
                                            density=72, 
                                            scales=[1],
                                            pickle_path=pickle_path)
    
    save_path = pickle_path.split("/")[-1].replace(".pickle", "")
    save_path_std = f"./images/std_{save_path}"
    save_path_composite = f"./images/composite_{save_path}"
    DensityOfStates.plot_std_dev_over_time(loadpath=pickle_path, 
                                           save_path=save_path_std, 
                                             sweep_list=[1,0.95,0.90,0.85,0.80],
                                             scales=[1],
                                             max_time_step=100)
    
    DensityOfStates.plot_composite_visualization(loadpath=pickle_path,
                                                save_path=save_path_composite,
                                                sweep_list=[1,0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50],
                                                scales=[1],
                                                max_time_step=200)
    ##############################################################################
    
    ################# Used for generating the synthetic dataset #################
    """
    NOTE: For generating dataset with different initial conditions, in the ./netlogo_models/forest_fire_evolution_stochastic_heterogenous_64grid.nlogo
    1. Comment line 36-40, and uncomment line 42-44 for random patch configurations
    2. Comment line 49-53, and uncomment line 55-57 for random fire seed locations
    """
    # Generate the sameInit and the diffInit dataset for different stochasticity levels
    dataset_generation.generate_stochastic_dataset()
    ##############################################################################
    
    
if __name__ == "__main__":
    main() 

