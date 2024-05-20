import yaml
import subprocess
from easydict import EasyDict
from utils.config_utils import load_config, save_config, merge_configs, dict_recursive
import os

def modify_yaml_and_run(yaml_file_path, experiment_list, eval_only=False):
    """ 
    If eval_only is True, then base_config is loaded from yaml_file_path and update_config is loaded from experiment_list.
    Otherwise, base_config is loaded from the default config file and update_config is loaded from experiment_list.
    """
    for experiment_name, parameters in experiment_list.items():
        print(f" ------- Running experiment: {experiment_name} ------- ")
        
        # Create an update configuration with modified parameters
        update_config = EasyDict()
        update_config["experiment_setting"] = {"experiment_name": experiment_name}
        for parameter_path, value in parameters.items():
            keys = parameter_path.split('.')
            temp_content = update_config
            for key in keys[:-1]:
                if key not in temp_content:
                    temp_content[key] = {}
                temp_content = temp_content[key]
            temp_content[keys[-1]] = value

        # Run the main.py script with the merged config
        if eval_only:
            base_config = load_config(yaml_file_path)
            os.makedirs("./config/eval", exist_ok=True)
            save_config(dict_recursive(base_config), './config/eval/base_config.yaml') 
            save_config(dict_recursive(update_config), './config/eval/update_config.yaml') 
            
            subprocess.run(["python", "main.py", 
                            "--base_config", "./config/eval/base_config.yaml", 
                            "--update_config", "./config/eval/update_config.yaml"])
        else:
            save_config(dict_recursive(update_config), './config/update_config.yaml') 
            subprocess.run(["python", "main.py"])

def list_to_string(lst):
    return "_".join(map(str, lst))

class MultiScalev2:
    @staticmethod
    def multiscale_ablatation(experiment_name_prefix, bce_weight_list, bottleneck_channel_list, bottleneck_flag_list, 
                              kernel_size_list, multilayer_only=False, model_name="d_convlstm", DC_filter_train=False):
        experiment_list = {}

        for bottleneck_channel in bottleneck_channel_list:
            for kernel_size in kernel_size_list:
                for bottleneck_flag in bottleneck_flag_list:    
                    for bce_weight in bce_weight_list:
                        # experiment_name = f"{experiment_name_prefix}__bceWeight{bce_weight}_bottleneck{bottleneck_flag}_kernelSize{kernel_size}_bottleneckChannel{bottleneck_channel}"
                        experiment_name = f"{experiment_name_prefix}__bottleneck{bottleneck_flag}_kernelSize{kernel_size}_bottleneckChannel{bottleneck_channel}"
                        
                        # Setting resume_from_checkpoint and load_only_model to False (ensuring default behavior)
                        resume_from_checkpoint = False
                        load_only_model = False
                        checkpoint_path = "null"
                        
                        
                        experiment_list[experiment_name] = {
                            "experiment_setting.train.resume": resume_from_checkpoint,
                            "experiment_setting.train.load_only_model": load_only_model,
                            "experiment_setting.train.checkpoint_path": checkpoint_path,
                            "experiment_setting.logs": True,    
                            # "model.d_convLSTM.bottleneck_factor": bottleneck_factor,
                            "model.segmentation_model.use_bottleneck": bottleneck_flag,
                            "model.name": model_name,
                            "experiment_setting.train.weight_for_minority_class": bce_weight,
                            "model.segmentation_model.conv_kernel_size": kernel_size,
                            "model.segmentation_model.bottleneck_channel": bottleneck_channel,
                            }
                        
                        if multilayer_only: 
                            # Modifications for multilayer experiments (No pyramid, only layers)
                            # experiment_list[experiment_name]["model.d_convLSTM_MS.ConvLSTM.layer_scales"] = [1,1,1,1]
                            experiment_list[experiment_name]["model.d_convLSTM_MS.ConvLSTM.layer_scales"] = [1,1]
                            experiment_list[experiment_name]["model.d_convLSTM_MS.ConvLSTM.pooling_type"] = "none"
                        
                        if DC_filter_train:
                            experiment_list[experiment_name]["experiment_setting.input_filtering.use"] = True
                            experiment_list[experiment_name]["experiment_setting.input_filtering.dc_threshold"] = 0.1
                            
                            
                            
                
        return experiment_list
        
   

def multiscale_ablatation_same_train_eval(experiment_name_prefix, multilayer_only=False, model_name="d_convlstm", channel=16, DC_filter_train=False):
    # bce_weight_list = [1,5,10,20,40,60]
    bce_weight_list = [3]
    
    # bottleneck_flag_list = [False, True]
    bottleneck_flag_list = [True]
    
    # kernel_size_list = [5,7,3]
    kernel_size_list = [3]
    
    # bottleneck_channel_list = [16,32]
    # bottleneck_channel_list = [16]
    bottleneck_channel_list = [channel]
    return MultiScalev2.multiscale_ablatation(experiment_name_prefix=experiment_name_prefix,
                                              bce_weight_list=bce_weight_list,
                                              multilayer_only=multilayer_only,
                                              model_name=model_name,
                                              bottleneck_flag_list=bottleneck_flag_list,
                                              kernel_size_list=kernel_size_list,
                                              bottleneck_channel_list=bottleneck_channel_list,
                                              DC_filter_train=DC_filter_train)
    

class EvaluationJobs:
    def __init__(self, jobname_list) -> None:
        """
        Args:
            jobname_list (list): List of job names to evaluate
        """
        self.job_name_list = jobname_list
    
    @staticmethod
    def get_yaml_filepath(job_name):
        return f"./artifacts/{job_name}/config.yaml"
        
    @staticmethod
    def generate_evaluation_only_modification(job_name, experiment_name_prefix, save_logs=True):
        """
        Creates a dictionary of parameters to modify in the yaml file for evaluation only.
        args:
            job_name (str): Name of the job to evaluate
            experiment_name_prefix (str): Prefix to append to the experiment name
            save_logs (bool): Whether to save logs
        """
        experiment_list = {}
        experiment_name = f"{experiment_name_prefix}_{job_name}" 
        model_weights_path = f"./artifacts/{job_name}"   
        
        experiment_list[experiment_name] = {
            "experiment_setting.evaluation.eval_only": True,
            "experiment_setting.evaluation.model_weights_path": model_weights_path,
            "experiment_setting.train.load_only_model": True,
            "experiment_setting.logs": save_logs,
            # "dataloader.test_batch_size": 1,
            "dataloader.device": "cuda:0",
            }
           
        return experiment_list

    def launch_evaluation_jobs(self, experiment_name_prefix, save_logs=True):
        for jobname in self.job_name_list:
            experiment_list = EvaluationJobs.generate_evaluation_only_modification(jobname,
                                                                                   experiment_name_prefix=experiment_name_prefix,
                                                                                   save_logs=save_logs)
                                                                            
            yaml_filepath = EvaluationJobs.get_yaml_filepath(jobname)
            modify_yaml_and_run(yaml_filepath, experiment_list, eval_only=True)
    
if __name__ == "__main__": 
    yaml_file_path = "./config/base_config.yaml"    
    
    # ##################### BCE Weight Sweep #####################
    # experiment_name_prefix = "d_convlstm_bce_sweep"
    # experiment_list_multiscale_ablation_same_train_eval1 = multiscale_ablatation_same_train_eval(experiment_name_prefix=experiment_name_prefix, 
    #                                                                                              multilayer_only=False, 
    #                                                                                              model_name="d_convlstm")
    # modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_ablation_same_train_eval1)
    # ############################################################
    
    # ##################### Segmentation model bottleneck #####################
    # experiment_name_prefix = "segmentation_model_1000epochs"
    # experiment_list_multiscale_ablation_same_train_eval1 = multiscale_ablatation_same_train_eval(experiment_name_prefix=experiment_name_prefix, 
    #                                                                                              model_name="segmentation_model")
    # modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_ablation_same_train_eval1)
    # #########################################################################
    
    # ##################### Segmentation model bottleneck #####################
    # experiment_name_prefix = "segmentation_model_1000epochs_kernelSweep"
    # experiment_list_multiscale_ablation_same_train_eval1 = multiscale_ablatation_same_train_eval(experiment_name_prefix=experiment_name_prefix, 
    #                                                                                              multilayer_only=False, 
    #                                                                                              model_name="segmentation_model")
    # modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_ablation_same_train_eval1)
    # #########################################################################
    
    # ##################### Segmentation model deep-bottleneck #####################
    # experiment_name_prefix = "segmentation_model_500epochs_deepBottleneck"
    # experiment_list_multiscale_ablation_same_train_eval1 = multiscale_ablatation_same_train_eval(experiment_name_prefix=experiment_name_prefix, 
    #                                                                                              multilayer_only=False, 
    #                                                                                              model_name="segmentation_model")
    # modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_ablation_same_train_eval1)
    # ##############################################################################
    
    # ##################### Segmentation model MC runs #############################
    # # channel 16
    # for i in range(10,30):
    #     experiment_name_prefix = f"segmentation_model_300epochs_MC_run{i}"
    #     experiment_list_multiscale_ablation_same_train_eval1 = multiscale_ablatation_same_train_eval(experiment_name_prefix=experiment_name_prefix, 
    #                                                                                                 multilayer_only=False, 
    #                                                                                                 model_name="segmentation_model",
    #                                                                                                 channel=16)
    #     modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_ablation_same_train_eval1)
    
    # # channel 32
    # for i in range(0,30):
    #     experiment_name_prefix = f"segmentation_model_300epochs_MC_run{i}"
    #     experiment_list_multiscale_ablation_same_train_eval1 = multiscale_ablatation_same_train_eval(experiment_name_prefix=experiment_name_prefix, 
    #                                                                                                 multilayer_only=False, 
    #                                                                                                 model_name="segmentation_model",
    #                                                                                                 channel=32)
    #     modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_ablation_same_train_eval1)
    # ##############################################################################
    
    # ##################### Segmentation model bottleneck -> DC Filter experiment #####################
    # experiment_name_prefix = "segmentation_model_DC_filteredTrue"
    # experiment_list_multiscale_ablation_same_train_eval1 = multiscale_ablatation_same_train_eval(experiment_name_prefix=experiment_name_prefix, 
    #                                                                                              model_name="segmentation_model",
    #                                                                                              DC_filter_train=True)
    # modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_ablation_same_train_eval1)
    # #########################################################################
    
    # # ##################### Segmentation model bottleneck -> DC Filter experiment | MC runs ############################
    # for i in range(0,30):
    #     experiment_name_prefix = f"segmentation_model_300epochs_MC_run{i}_DC_filteredTrue"
    #     experiment_list_multiscale_ablation_same_train_eval1 = multiscale_ablatation_same_train_eval(experiment_name_prefix=experiment_name_prefix, 
    #                                                                                                 model_name="segmentation_model",
    #                                                                                                 DC_filter_train=True)
    #     modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_ablation_same_train_eval1)
        
    # # Evaluation job list
    # jobname_list = []
    # for i in range(0,30):
    #     experiment_name_bottleneck = f"segmentation_model_300epochs_MC_run{i}_DC_filteredTrue__bottleneckTrue_kernelSize3_bottleneckChannel16"
    #     jobname_list.append(experiment_name_bottleneck)
    # ###################################################################################################################
   
    
   
    ######### Evaluation Jobs #########
    # # jobname list for the MC runs
    # jobname_list = []
    # for i in range(1,2):
    #     experiment_name_bottleneck16 = f"segmentation_model_300epochs_MC_run{i}__bottleneckTrue_kernelSize3_bottleneckChannel16"
    #     experiment_name_bottleneck32 = f"segmentation_model_300epochs_MC_run{i}__bottleneckTrue_kernelSize3_bottleneckChannel32"
    #     experiment_name_NObottleneck16 = f"segmentation_model_300epochs_MC_run{i}__bottleneckFalse_kernelSize3_bottleneckChannel16"
    #     experiment_name_NObottleneck32 = f"segmentation_model_300epochs_MC_run{i}__bottleneckFalse_kernelSize3_bottleneckChannel32"
    #     jobname_list.append(experiment_name_NObottleneck32)
    #     break
    #     jobname_list.append(experiment_name_bottleneck16)
    #     jobname_list.append(experiment_name_NObottleneck16)
    #     jobname_list.append(experiment_name_bottleneck32)
    
    jobname_list = [
                    # "d_convlstm_bce_sweep__bceWeight1",
                    # "d_convlstm_bce_sweep__bceWeight5",
                    # "d_convlstm_bce_sweep__bceWeight10",
                    # "d_convlstm_bce_sweep__bceWeight20",
                    # "d_convlstm_bce_sweep__bceWeight40",
                    # "d_convlstm_bce_sweep__bceWeight60"
                    
                    # "segmentation_model__bceWeight3_bottleneckTrue",
                    # "segmentation_model__bceWeight3_bottleneckFalse",
                    
                    # "segmentation_model_1000epochs__bceWeight3_bottleneckTrue",
                    # "segmentation_model_1000epochs__bceWeight3_bottleneckFalse",
                    
                    # "segmentation_model_1000epochs_kernelSweep__bceWeight3_bottleneckTrue_kernelSize3",
                    # "segmentation_model_1000epochs_kernelSweep__bceWeight3_bottleneckFalse_kernelSize3",
                    # "segmentation_model_1000epochs_kernelSweep__bceWeight3_bottleneckTrue_kernelSize5",
                    # "segmentation_model_1000epochs_kernelSweep__bceWeight3_bottleneckFalse_kernelSize5",
                    # "segmentation_model_1000epochs_kernelSweep__bceWeight3_bottleneckTrue_kernelSize7",
                    # "segmentation_model_1000epochs_kernelSweep__bceWeight3_bottleneckFalse_kernelSize7",
                    
                    # "segmentation_model_500epochs_deepBottleneck__bottleneckTrue_kernelSize3_bottleneckChannel16",
                    # "segmentation_model_500epochs_deepBottleneck__bottleneckFalse_kernelSize3_bottleneckChannel16",
                    # "segmentation_model_500epochs_deepBottleneck__bottleneckTrue_kernelSize3_bottleneckChannel32",
                    # "segmentation_model_500epochs_deepBottleneck__bottleneckFalse_kernelSize3_bottleneckChannel32",
                    
                    # # DC Filter experiment
                    # "segmentation_model_DC_filteredTrue__bottleneckTrue_kernelSize3_bottleneckChannel16"
                    
                    # MC runs of interest for expected calibration error
                    "segmentation_model_300epochs_MC_run10__bottleneckTrue_kernelSize3_bottleneckChannel16",    
                    # "segmentation_model_300epochs_MC_run28__bottleneckFalse_kernelSize3_bottleneckChannel16",    
                    
                    ]
                    
                    
                    
    evaluation_jobs = EvaluationJobs(jobname_list)
    evaluation_jobs.launch_evaluation_jobs(save_logs=False, 
                                           experiment_name_prefix="eval_ECE")
    