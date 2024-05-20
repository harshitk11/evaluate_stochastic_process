import yaml
import subprocess
from easydict import EasyDict
from utils.config_utils import load_config, save_config, merge_configs, dict_recursive

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
    def multiscale_ablatation(densities, stochasticities, train_scales, eval_scales, bottleneck_factor_list, experiment_name_prefix, learnable_scale_weights=False, multilayer_only=False, model_name="d_convlstm"):
        experiment_list = {}
    
        for density in densities:
            for stochasticity in stochasticities:
                for train_scale in train_scales:
                    for eval_scale in eval_scales:
                        for bottleneck_factor in bottleneck_factor_list:
                            experiment_name = f"{experiment_name_prefix}__d{density}_p{stochasticity}_b{bottleneck_factor}_train_{list_to_string(train_scale)}_eval_{list_to_string(eval_scale)}"
                            
                            # Setting resume_from_checkpoint and load_only_model to False (ensuring default behavior)
                            resume_from_checkpoint = True
                            load_only_model = False
                            checkpoint_path = "null"
                            
                            dataset_name = f"density{density}_p_{stochasticity}_uniform"
                            
                            experiment_list[experiment_name] = {
                                "experiment_setting.train.dataset_name": dataset_name,
                                "experiment_setting.evaluation.dataset_name": dataset_name,
                                "experiment_setting.train.multiscale.loss_calculation_scales": train_scale,
                                "experiment_setting.evaluation.multiscale.loss_calculation_scales": eval_scale,
                                "experiment_setting.train.resume": resume_from_checkpoint,
                                "experiment_setting.train.load_only_model": load_only_model,
                                "experiment_setting.train.checkpoint_path": checkpoint_path,
                                "experiment_setting.train.multiscale.learned_multiscale_weights": learnable_scale_weights,
                                "experiment_setting.logs": True,    
                                
                                "model.d_convLSTM.bottleneck_factor": bottleneck_factor,
                                "model.d_convLSTM_MS.bottleneck_factor": bottleneck_factor,
                                
                                "model.name": model_name,
                                }
                           
                            if multilayer_only: 
                                # Modifications for multilayer experiments (No pyramid, only layers)
                                # experiment_list[experiment_name]["model.d_convLSTM_MS.ConvLSTM.layer_scales"] = [1,1,1,1]
                                experiment_list[experiment_name]["model.d_convLSTM_MS.ConvLSTM.layer_scales"] = [1,1]
                                experiment_list[experiment_name]["model.d_convLSTM_MS.ConvLSTM.pooling_type"] = "none"
                
        return experiment_list
        
    
    @staticmethod
    def multiscale_sequential(densities, train_scales, eval_scales):
        experiment_list = {}

        experiment_name_prefix = "multiscale_sequential"
        checkpoint_paths = {
            4: f"{experiment_name_prefix}__forest_{{density}}_train_8_eval_1",
            2: f"{experiment_name_prefix}__forest_{{density}}_train_4_eval_1",
            1: f"{experiment_name_prefix}__forest_{{density}}_train_2_eval_1",
        }

        for density in densities:
            for train_scale in train_scales:
                for eval_scale in eval_scales:
                    scale = train_scale[0]
                    experiment_name = f"{experiment_name_prefix}__forest_{density}_train_{list_to_string(train_scale)}_eval_{list_to_string(eval_scale)}"
                    
                    resume_from_checkpoint = load_only_model = scale in {1, 2, 4}
                    checkpoint_path = "null" if scale == 8 else f"./artifacts/{checkpoint_paths[scale].format(density=density)}"
                    
                    experiment_list[experiment_name] = {
                        "experiment_setting.train.dataset_name": f"density{density}_uniform",
                        "experiment_setting.evaluation.dataset_name": f"density{density}_uniform",
                        "experiment_setting.train.multiscale.loss_calculation_scales": train_scale,
                        "experiment_setting.evaluation.multiscale.loss_calculation_scales": eval_scale,
                        "experiment_setting.train.resume": resume_from_checkpoint,
                        "experiment_setting.train.load_only_model": load_only_model,
                        "experiment_setting.train.checkpoint_path": checkpoint_path,
                        "experiment_setting.logs": True,    
                    }
                    
        return experiment_list

def multiscale_ablatation_same_train_eval(scale, experiment_name_prefix, multilayer_only=False, model_name="d_convlstm"):
    bottleneck_factor_list = [1] 
    # bottleneck_factor_list = [1,2,4,8] 
    # bottleneck_factor_list = [8] 
    
    densities = [72]
    
    # stochasticities = [100]
    # stochasticities = [100, 95, 90, 85, 80]
    stochasticities = [80]
    
    if scale == 8:
        train_scales = [[8]]
        eval_scales = [[8]]
    elif scale == 4:
        train_scales = [[4]]
        eval_scales = [[4]]
    elif scale == 2:
        train_scales = [[2]]
        eval_scales = [[2]]
    elif scale == 1:
        train_scales = [[1]]
        eval_scales = [[1]]
    else:
        raise ValueError("Invalid scale")
    return MultiScalev2.multiscale_ablatation(densities, stochasticities, train_scales, eval_scales, bottleneck_factor_list, 
                                              experiment_name_prefix=experiment_name_prefix, 
                                              multilayer_only=multilayer_only,
                                              model_name=model_name)
    
            
def all_multiclass_experiment(experiment_name_prefix, multilayer_only=False, model_name="d_convlstm"):
    densities = [72]
    stochasticities = [100, 95, 90, 85, 80]
    bottleneck_factor_list = [1]
    
    # train_scales = [[1,8], [1,4], [1,2], [1,2,4], [1,2,8], [1,4,8], [1,2,4,8]]
    # train_scales = [[1,2], [1,2,4], [1,2,4,8]]
    train_scales = [[1,2]]
    
    eval_scales = [[1]]
    
    return MultiScalev2.multiscale_ablatation(densities, stochasticities, train_scales, eval_scales, bottleneck_factor_list, 
                                              experiment_name_prefix=experiment_name_prefix,
                                              multilayer_only=multilayer_only,
                                              model_name=model_name)

def multiscale_sequential():
    densities = [76, 68]
    train_scales = [[8], [4], [2], [1]]
    eval_scales = [[1]]
    return MultiScalev2.multiscale_sequential(densities, train_scales, eval_scales)

def learnable_scale_weights():
    densities = [76, 68]
    train_scales = [[1,2,4,8]]
    eval_scales = [[1]]
    return MultiScalev2.multiscale_ablatation(densities, train_scales, eval_scales, experiment_name_prefix="learnable_scale_weights_evalLossFixed", learnable_scale_weights=True)

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
    def generate_evaluation_only_modification(job_name, experiment_name_prefix, dataset_selector, save_logs=True, eval_dataset_name=None):
        """
        Creates a dictionary of parameters to modify in the yaml file for evaluation only.
        args:
            job_name (str): Name of the job to evaluate
            experiment_name_prefix (str): Prefix to append to the experiment name
            save_logs (bool): Whether to save logs
            eval_dataset_name (str): Name of the dataset to evaluate on. If None, then the dataset name is determined from the job name.
            dataset_selector (str): Selects the dataset type to evaluate on. 
                                    ["sameInit", "diffInit"]
        """
        experiment_list = {}
        experiment_name = f"{experiment_name_prefix}_{job_name}" 
        model_weights_path = f"./artifacts/{job_name}"   
        
        dataset_mapper = {
                "sameInit": {"p_100": "density72_p_100_sameInit_1", 
                             "p_95": "density72_p_95_sameInit_1",
                             "p_90": "density72_p_90_sameInit_1",
                             "p_85": "density72_p_85_sameInit_1",
                             "p_80": "density72_p_80_sameInit_1"},
                "diffInit": {"p_100": "density72_p_100_uniform",
                             "p_95": "density72_p_95_uniform",
                            "p_90": "density72_p_90_uniform",
                            "p_85": "density72_p_85_uniform",
                            "p_80": "density72_p_80_uniform"},
        }
        datasetSelector = dataset_mapper[dataset_selector]
        
        if "p100" in job_name:
            mapped_eval_dataset_name = datasetSelector["p_100"]
        elif "p95" in job_name:
            mapped_eval_dataset_name = datasetSelector["p_95"]
        elif "p90" in job_name:
            mapped_eval_dataset_name = datasetSelector["p_90"]
        elif "p85" in job_name:
            mapped_eval_dataset_name = datasetSelector["p_85"]
        elif "p80" in job_name:
            mapped_eval_dataset_name = datasetSelector["p_80"]
        else:
            raise ValueError("Invalid job name")
        
        
        experiment_list[experiment_name] = {
            # "experiment_setting.evaluation.multiscale.loss_calculation_scales": [1],
            "experiment_setting.evaluation.eval_only": True,
            "experiment_setting.evaluation.model_weights_path": model_weights_path,
            "experiment_setting.train.load_only_model": True,
            "experiment_setting.logs": save_logs,
            "dataloader.test_batch_size": 1,
            "dataloader.device": "cuda:0",
            # "dataloader.train_val_split": [0,1.0]
            }
        
        if eval_dataset_name is not None:
            experiment_list[experiment_name]["experiment_setting.evaluation.dataset_name"] = eval_dataset_name
        else: 
            experiment_list[experiment_name]["experiment_setting.evaluation.dataset_name"] = mapped_eval_dataset_name
        
        print(f"Eval dataset name: {experiment_list[experiment_name]['experiment_setting.evaluation.dataset_name']}")
        return experiment_list

    def launch_evaluation_jobs(self, experiment_name_prefix, dataset_selector, save_logs=True, eval_dataset_name=None):
        for jobname in self.job_name_list:
            experiment_list = EvaluationJobs.generate_evaluation_only_modification(jobname,
                                                                                   experiment_name_prefix=experiment_name_prefix,
                                                                                   dataset_selector=dataset_selector, 
                                                                                   save_logs=save_logs, 
                                                                                   eval_dataset_name=eval_dataset_name)
            yaml_filepath = EvaluationJobs.get_yaml_filepath(jobname)
            modify_yaml_and_run(yaml_filepath, experiment_list, eval_only=True)
    
if __name__ == "__main__": 
    yaml_file_path = "./config/base_config.yaml"
    # # Supervision from subsets of all scales | Evaluate at scale 1
    # # experiment_list_all_multiclass = all_multiclass_experiment(experiment_name_prefix="pyramid_v2_stochastic_multiscale_ablation")
    # experiment_list_all_multiclass = all_multiclass_experiment(experiment_name_prefix="pyramid_v2_stochastic_multiscale")
    # modify_yaml_and_run(yaml_file_path, experiment_list_all_multiclass)
    
    # Train and evaluate at same scale
    # experiment_name_prefix = "pyramid_v2_stochastic_multiscale_ablation"
    # experiment_name_prefix = "1_multilayerPyramidLSTM_predPoolMax_lstmPoolMax_stochastic_ablation"
    # experiment_name_prefix = "pyramid_v2_stochastic_multiscale_ablation_bottleNeck"
    
    # experiment_list_multiscale_ablation_same_train_eval8 = multiscale_ablatation_same_train_eval(scale=8, experiment_name_prefix=experiment_name_prefix)
    # experiment_list_multiscale_ablation_same_train_eval4 = multiscale_ablatation_same_train_eval(scale=4, experiment_name_prefix=experiment_name_prefix)
    # experiment_list_multiscale_ablation_same_train_eval2 = multiscale_ablatation_same_train_eval(scale=2, experiment_name_prefix=experiment_name_prefix)
    # experiment_list_multiscale_ablation_same_train_eval1 = multiscale_ablatation_same_train_eval(scale=1, experiment_name_prefix=experiment_name_prefix, multiscale_multilayer=True)
    # modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_ablation_same_train_eval8)
    # modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_ablation_same_train_eval4)
    # modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_ablation_same_train_eval2)
    # modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_ablation_same_train_eval1)
    
    
    # ###################################### Bottleneck experiments ######################################
    # experiment_name_prefix = "7_pyramid_v2_stochastic_multiscale_ablation_bottleNeck"
    # experiment_list_multiscale_ablation_same_train_eval1 = multiscale_ablatation_same_train_eval(scale=1, 
    #                                                                                              experiment_name_prefix=experiment_name_prefix, 
    #                                                                                              multilayer_only=False,
    #                                                                                              model_name="d_convlstm")
    # modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_ablation_same_train_eval1)
    # ####################################################################################################
    
    # ###################################### Multilayer experiments ######################################
    # experiment_name_prefix = "2_multilayerLSTM_predPoolMax_stochastic"
    # experiment_list_multiscale_ablation_same_train_eval1 = multiscale_ablatation_same_train_eval(scale=1, 
    #                                                                                              experiment_name_prefix=experiment_name_prefix, 
    #                                                                                              multilayer_only=True,
    #                                                                                              model_name="depth_convlstm_multiscale")
    # modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_ablation_same_train_eval1)
    
    # experiment_name_prefix = "1_multilayerPyramidLSTM_predPoolMax_lstmPoolMax_stochastic"
    # experiment_list_multiscale_ablation_same_train_eval1 = multiscale_ablatation_same_train_eval(scale=1, 
    #                                                                                              experiment_name_prefix=experiment_name_prefix, 
    #                                                                                              multilayer_only=False,
    #                                                                                              model_name="depth_convlstm_multiscale")
    # modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_ablation_same_train_eval1)
    # ####################################################################################################
    
    ###################################### Multilayer experiments with multiscale loss ######################################
    # experiment_name_prefix = "3_multilayerLSTM_multiScaleLoss_predPoolMax_stochastic"
    # experiment_list_multiscale_ablation_same_train_eval1 = all_multiclass_experiment(experiment_name_prefix=experiment_name_prefix, 
    #                                                                                 multilayer_only=True,
    #                                                                                 model_name="depth_convlstm_multiscale")
    # modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_ablation_same_train_eval1)
    
    # experiment_name_prefix = "4_multilayerPyramidLSTM_multiScaleLoss_predPoolMax_lstmPoolMax_stochastic"
    # experiment_list_multiscale_ablation_same_train_eval1 = all_multiclass_experiment(experiment_name_prefix=experiment_name_prefix, 
    #                                                                                 multilayer_only=False,
    #                                                                                 model_name="depth_convlstm_multiscale")
    # modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_ablation_same_train_eval1)
    
    # experiment_name_prefix = "5_multilayerLSTM_1_2_multiScaleLoss_predPoolMax_stochastic"
    # experiment_list_multiscale_ablation_same_train_eval1 = all_multiclass_experiment(experiment_name_prefix=experiment_name_prefix, 
    #                                                                                 multilayer_only=True,
    #                                                                                 model_name="depth_convlstm_multiscale")
    # modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_ablation_same_train_eval1)
    
    # experiment_name_prefix = "6_multilayerPyramidLSTM_1_2_multiScaleLoss_predPoolMax_lstmPoolMax_stochastic"
    # experiment_list_multiscale_ablation_same_train_eval1 = all_multiclass_experiment(experiment_name_prefix=experiment_name_prefix, 
    #                                                                                 multilayer_only=False,
    #                                                                                 model_name="depth_convlstm_multiscale")
    # modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_ablation_same_train_eval1)
    #########################################################################################################################   
    
    # ###################################### Bottleneck study ############################################
    # # Without multilayer ConvLSTM
    # experiment_name_prefix = "8_pyramid_v2_bottleneckStudy"
    # experiment_list_multiscale_ablation_same_train_eval1 = multiscale_ablatation_same_train_eval(scale=1, 
    #                                                                                              experiment_name_prefix=experiment_name_prefix, 
    #                                                                                              multilayer_only=False,
    #                                                                                              model_name="d_convlstm")
    # modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_ablation_same_train_eval1)
    
    # # With multilayer ConvLSTM
    # experiment_name_prefix = "9_multilayerLSTM_bottleneckStudy"
    # experiment_list_multiscale_ablation_same_train_eval1 = multiscale_ablatation_same_train_eval(scale=1, 
    #                                                                                              experiment_name_prefix=experiment_name_prefix, 
    #                                                                                              multilayer_only=True,
    #                                                                                              model_name="depth_convlstm_multiscale")
    # modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_ablation_same_train_eval1)
    # ####################################################################################################
    
    # # Train at different scales and evaluate at scale 1
    # experiment_list_multiscale_ablation = multiscale_ablatation(experiment_name_prefix="pyramid_v2_revWeight_meanPool_multiscale_ablation")
    # modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_ablation)
    
    # experiment_list_multiscale_sequential = multiscale_sequential()
    # modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_sequential)
    
    # experiment_learnable_scale_weights = learnable_scale_weights()
    # modify_yaml_and_run(yaml_file_path, experiment_learnable_scale_weights)
    
    ## For debugging
    # for k,v in experiment_list_no_resume.items():
    #     print(f"-------------------- {k} --------------------")
    #     for k2,v2 in v.items():
    #         print(k2, v2)
   
    # ####### For training ARNCA model ########
    # experiment_name_prefix = "arnca"
    # experiment_list_multiscale_ablation_same_train_eval1 = multiscale_ablatation_same_train_eval(scale=1, 
    #                                                                                              experiment_name_prefix=experiment_name_prefix, 
    #                                                                                              multilayer_only=True,
    #                                                                                              model_name="arnca")
    # modify_yaml_and_run(yaml_file_path, experiment_list_multiscale_ablation_same_train_eval1)
    # exit()
    # #########################################
   
    ######### Evaluation Jobs #########
    jobname_list = [
                    # "pyramid_v2_stochastic_multiscale_ablation__d72_p100_train_1_eval_1",
                    # "pyramid_v2_stochastic_multiscale_ablation__d72_p95_train_1_eval_1",
                    # "pyramid_v2_stochastic_multiscale_ablation__d72_p90_train_1_eval_1",
                    # "pyramid_v2_stochastic_multiscale_ablation__d72_p85_train_1_eval_1",
                    # "pyramid_v2_stochastic_multiscale_ablation__d72_p80_train_1_eval_1",
                    
                    # "pyramid_v2_stochastic_multiscale_ablation__d72_p100_train_2_eval_2",
                    # "pyramid_v2_stochastic_multiscale_ablation__d72_p95_train_2_eval_2",
                    # "pyramid_v2_stochastic_multiscale_ablation__d72_p90_train_2_eval_2",
                    # "pyramid_v2_stochastic_multiscale_ablation__d72_p85_train_2_eval_2",
                    # "pyramid_v2_stochastic_multiscale_ablation__d72_p80_train_2_eval_2",
                    
                    # "pyramid_v2_stochastic_multiscale_ablation__d72_p100_train_4_eval_4",
                    # "pyramid_v2_stochastic_multiscale_ablation__d72_p95_train_4_eval_4",
                    # "pyramid_v2_stochastic_multiscale_ablation__d72_p90_train_4_eval_4",
                    # "pyramid_v2_stochastic_multiscale_ablation__d72_p85_train_4_eval_4",
                    # "pyramid_v2_stochastic_multiscale_ablation__d72_p80_train_4_eval_4",
                    
                    # "pyramid_v2_stochastic_multiscale_ablation__d72_p100_train_8_eval_8",
                    # "pyramid_v2_stochastic_multiscale_ablation__d72_p95_train_8_eval_8",
                    # "pyramid_v2_stochastic_multiscale_ablation__d72_p90_train_8_eval_8",
                    # "pyramid_v2_stochastic_multiscale_ablation__d72_p85_train_8_eval_8",
                    # "pyramid_v2_stochastic_multiscale_ablation__d72_p80_train_8_eval_8",
    
                    # "1_multilayerPyramidLSTM_predPoolMax_lstmPoolMax_stochastic_ablation__d72_p100_train_1_eval_1",  
                    # "1_multilayerPyramidLSTM_predPoolMax_lstmPoolMax_stochastic_ablation__d72_p95_train_1_eval_1",  
                    # "1_multilayerPyramidLSTM_predPoolMax_lstmPoolMax_stochastic_ablation__d72_p90_train_1_eval_1", 
                    # "1_multilayerPyramidLSTM_predPoolMax_lstmPoolMax_stochastic_ablation__d72_p85_train_1_eval_1", 
                    # "1_multilayerPyramidLSTM_predPoolMax_lstmPoolMax_stochastic_ablation__d72_p80_train_1_eval_1",  
                    
                    # "pyramid_v2_stochastic_multiscale_ablation_bottleNeck__d72_p100_b1_train_1_eval_1",
                    # "pyramid_v2_stochastic_multiscale_ablation_bottleNeck__d72_p100_b2_train_1_eval_1",
                    # "pyramid_v2_stochastic_multiscale_ablation_bottleNeck__d72_p100_b4_train_1_eval_1",
                    # "pyramid_v2_stochastic_multiscale_ablation_bottleNeck__d72_p100_b8_train_1_eval_1",
                    
                    # "pyramid_v2_stochastic_multiscale__d72_p100_b1_train_1_2_eval_1",
                    # "pyramid_v2_stochastic_multiscale__d72_p95_b1_train_1_2_eval_1",
                    # "pyramid_v2_stochastic_multiscale__d72_p90_b1_train_1_2_eval_1",
                    # "pyramid_v2_stochastic_multiscale__d72_p85_b1_train_1_2_eval_1",
                    # "pyramid_v2_stochastic_multiscale__d72_p80_b1_train_1_2_eval_1",
                    
                    # "pyramid_v2_stochastic_multiscale__d72_p100_b1_train_1_2_4_eval_1",
                    # "pyramid_v2_stochastic_multiscale__d72_p95_b1_train_1_2_4_eval_1",
                    # "pyramid_v2_stochastic_multiscale__d72_p90_b1_train_1_2_4_eval_1",
                    # "pyramid_v2_stochastic_multiscale__d72_p85_b1_train_1_2_4_eval_1",
                    # "pyramid_v2_stochastic_multiscale__d72_p80_b1_train_1_2_4_eval_1",
                    
                    # "pyramid_v2_stochastic_multiscale__d72_p100_b1_train_1_2_4_8_eval_1",
                    # "pyramid_v2_stochastic_multiscale__d72_p95_b1_train_1_2_4_8_eval_1",
                    # "pyramid_v2_stochastic_multiscale__d72_p90_b1_train_1_2_4_8_eval_1",
                    # "pyramid_v2_stochastic_multiscale__d72_p85_b1_train_1_2_4_8_eval_1",
                    # "pyramid_v2_stochastic_multiscale__d72_p80_b1_train_1_2_4_8_eval_1",
                    
                    # "1_multilayerPyramidLSTM_predPoolMax_lstmPoolMax_stochastic__d72_p100_b1_train_1_eval_1",
                    # "1_multilayerPyramidLSTM_predPoolMax_lstmPoolMax_stochastic__d72_p95_b1_train_1_eval_1",
                    # "1_multilayerPyramidLSTM_predPoolMax_lstmPoolMax_stochastic__d72_p90_b1_train_1_eval_1",
                    # "1_multilayerPyramidLSTM_predPoolMax_lstmPoolMax_stochastic__d72_p85_b1_train_1_eval_1",
                    # "1_multilayerPyramidLSTM_predPoolMax_lstmPoolMax_stochastic__d72_p80_b1_train_1_eval_1",
                    
                    # "2_multilayerLSTM_predPoolMax_stochastic__d72_p100_b1_train_1_eval_1",
                    # "2_multilayerLSTM_predPoolMax_stochastic__d72_p95_b1_train_1_eval_1",
                    # "2_multilayerLSTM_predPoolMax_stochastic__d72_p90_b1_train_1_eval_1",
                    # "2_multilayerLSTM_predPoolMax_stochastic__d72_p85_b1_train_1_eval_1",
                    # "2_multilayerLSTM_predPoolMax_stochastic__d72_p80_b1_train_1_eval_1",
                    
                    # "3_multilayerLSTM_multiScaleLoss_predPoolMax_stochastic__d72_p100_b1_train_1_2_eval_1",
                    # "3_multilayerLSTM_multiScaleLoss_predPoolMax_stochastic__d72_p95_b1_train_1_2_eval_1",
                    # "3_multilayerLSTM_multiScaleLoss_predPoolMax_stochastic__d72_p90_b1_train_1_2_eval_1",
                    # "3_multilayerLSTM_multiScaleLoss_predPoolMax_stochastic__d72_p85_b1_train_1_2_eval_1",
                    # "3_multilayerLSTM_multiScaleLoss_predPoolMax_stochastic__d72_p80_b1_train_1_2_eval_1",
                    
                    # "4_multilayerPyramidLSTM_multiScaleLoss_predPoolMax_lstmPoolMax_stochastic__d72_p100_b1_train_1_2_eval_1",
                    # "4_multilayerPyramidLSTM_multiScaleLoss_predPoolMax_lstmPoolMax_stochastic__d72_p95_b1_train_1_2_eval_1",
                    # "4_multilayerPyramidLSTM_multiScaleLoss_predPoolMax_lstmPoolMax_stochastic__d72_p90_b1_train_1_2_eval_1",
                    # "4_multilayerPyramidLSTM_multiScaleLoss_predPoolMax_lstmPoolMax_stochastic__d72_p85_b1_train_1_2_eval_1",
                    # "4_multilayerPyramidLSTM_multiScaleLoss_predPoolMax_lstmPoolMax_stochastic__d72_p80_b1_train_1_2_eval_1",
                    
                    # "5_multilayerLSTM_1_2_multiScaleLoss_predPoolMax_stochastic__d72_p100_b1_train_1_2_eval_1",
                    # "5_multilayerLSTM_1_2_multiScaleLoss_predPoolMax_stochastic__d72_p95_b1_train_1_2_eval_1",
                    # "5_multilayerLSTM_1_2_multiScaleLoss_predPoolMax_stochastic__d72_p90_b1_train_1_2_eval_1",
                    # "5_multilayerLSTM_1_2_multiScaleLoss_predPoolMax_stochastic__d72_p85_b1_train_1_2_eval_1",
                    # "5_multilayerLSTM_1_2_multiScaleLoss_predPoolMax_stochastic__d72_p80_b1_train_1_2_eval_1",
                    
                    # "6_multilayerPyramidLSTM_1_2_multiScaleLoss_predPoolMax_lstmPoolMax_stochastic__d72_p100_b1_train_1_2_eval_1",
                    # "6_multilayerPyramidLSTM_1_2_multiScaleLoss_predPoolMax_lstmPoolMax_stochastic__d72_p95_b1_train_1_2_eval_1",
                    # "6_multilayerPyramidLSTM_1_2_multiScaleLoss_predPoolMax_lstmPoolMax_stochastic__d72_p90_b1_train_1_2_eval_1",
                    # "6_multilayerPyramidLSTM_1_2_multiScaleLoss_predPoolMax_lstmPoolMax_stochastic__d72_p85_b1_train_1_2_eval_1",
                    # "6_multilayerPyramidLSTM_1_2_multiScaleLoss_predPoolMax_lstmPoolMax_stochastic__d72_p80_b1_train_1_2_eval_1",
                    
                    # "7_pyramid_v2_stochastic_multiscale_ablation_bottleNeck__d72_p100_b2_train_1_eval_1",
                    # "7_pyramid_v2_stochastic_multiscale_ablation_bottleNeck__d72_p95_b2_train_1_eval_1",
                    # "7_pyramid_v2_stochastic_multiscale_ablation_bottleNeck__d72_p90_b2_train_1_eval_1",
                    # "7_pyramid_v2_stochastic_multiscale_ablation_bottleNeck__d72_p85_b2_train_1_eval_1",
                    # "7_pyramid_v2_stochastic_multiscale_ablation_bottleNeck__d72_p80_b2_train_1_eval_1",
                    
                    # ########### Bottleneck study ############
                    # "8_pyramid_v2_bottleneckStudy__d72_p100_b1_train_1_eval_1",
                    # "8_pyramid_v2_bottleneckStudy__d72_p95_b1_train_1_eval_1",
                    # "8_pyramid_v2_bottleneckStudy__d72_p90_b1_train_1_eval_1",
                    # "8_pyramid_v2_bottleneckStudy__d72_p85_b1_train_1_eval_1",
                    # "8_pyramid_v2_bottleneckStudy__d72_p80_b1_train_1_eval_1",
                    
                    # "8_pyramid_v2_bottleneckStudy__d72_p100_b2_train_1_eval_1",
                    # "8_pyramid_v2_bottleneckStudy__d72_p95_b2_train_1_eval_1",
                    # "8_pyramid_v2_bottleneckStudy__d72_p90_b2_train_1_eval_1",
                    # "8_pyramid_v2_bottleneckStudy__d72_p85_b2_train_1_eval_1",
                    # "8_pyramid_v2_bottleneckStudy__d72_p80_b2_train_1_eval_1", # btk 2
                    
                    "8_pyramid_v2_bottleneckStudy__d72_p100_b4_train_1_eval_1",
                    "8_pyramid_v2_bottleneckStudy__d72_p95_b4_train_1_eval_1",
                    "8_pyramid_v2_bottleneckStudy__d72_p90_b4_train_1_eval_1",
                    "8_pyramid_v2_bottleneckStudy__d72_p85_b4_train_1_eval_1",
                    "8_pyramid_v2_bottleneckStudy__d72_p80_b4_train_1_eval_1", # btk 4
                    
                    # "8_pyramid_v2_bottleneckStudy__d72_p100_b8_train_1_eval_1",
                    # "8_pyramid_v2_bottleneckStudy__d72_p95_b8_train_1_eval_1",
                    # "8_pyramid_v2_bottleneckStudy__d72_p90_b8_train_1_eval_1",
                    # "8_pyramid_v2_bottleneckStudy__d72_p85_b8_train_1_eval_1",
                    # "8_pyramid_v2_bottleneckStudy__d72_p80_b8_train_1_eval_1",
                    
                    # "9_multilayerLSTM_bottleneckStudy__d72_p100_b1_train_1_eval_1",
                    # "9_multilayerLSTM_bottleneckStudy__d72_p95_b1_train_1_eval_1",
                    # "9_multilayerLSTM_bottleneckStudy__d72_p90_b1_train_1_eval_1",
                    # "9_multilayerLSTM_bottleneckStudy__d72_p85_b1_train_1_eval_1",
                    # "9_multilayerLSTM_bottleneckStudy__d72_p80_b1_train_1_eval_1", # multilayer LSTM
                    
                    # "9_multilayerLSTM_bottleneckStudy__d72_p100_b2_train_1_eval_1",
                    # "9_multilayerLSTM_bottleneckStudy__d72_p95_b2_train_1_eval_1",
                    # "9_multilayerLSTM_bottleneckStudy__d72_p90_b2_train_1_eval_1",
                    # "9_multilayerLSTM_bottleneckStudy__d72_p85_b2_train_1_eval_1",
                    # "9_multilayerLSTM_bottleneckStudy__d72_p80_b2_train_1_eval_1",
                    
                    # "9_multilayerLSTM_bottleneckStudy__d72_p100_b4_train_1_eval_1",
                    # "9_multilayerLSTM_bottleneckStudy__d72_p95_b4_train_1_eval_1",
                    # "9_multilayerLSTM_bottleneckStudy__d72_p90_b4_train_1_eval_1",
                    # "9_multilayerLSTM_bottleneckStudy__d72_p85_b4_train_1_eval_1",
                    # "9_multilayerLSTM_bottleneckStudy__d72_p80_b4_train_1_eval_1",
                    
                    # "9_multilayerLSTM_bottleneckStudy__d72_p100_b8_train_1_eval_1",
                    # "9_multilayerLSTM_bottleneckStudy__d72_p95_b8_train_1_eval_1",
                    # "9_multilayerLSTM_bottleneckStudy__d72_p90_b8_train_1_eval_1",
                    # "9_multilayerLSTM_bottleneckStudy__d72_p85_b8_train_1_eval_1",
                    # "9_multilayerLSTM_bottleneckStudy__d72_p80_b8_train_1_eval_1",
                    
                    # "arnca__d72_p100_b1_train_1_eval_1",
                    # "arnca__d72_p95_b1_train_1_eval_1",
                    # "arnca__d72_p90_b1_train_1_eval_1",
                    # "arnca__d72_p85_b1_train_1_eval_1",
                    # "arnca__d72_p80_b1_train_1_eval_1"
                    ]
                    
                    
                    
                    
    # datasetname_list = ["density72_p_100_sameInit_1", "density72_p_95_sameInit_1", "density72_p_90_sameInit_1", "density72_p_85_sameInit_1", "density72_p_80_sameInit_1"]
    evaluation_jobs = EvaluationJobs(jobname_list)
   
    # For evaluating on the same slevel as the training slevel
    # dataset_selector = "sameInit", "diffInit"
    # evaluation_jobs.launch_evaluation_jobs(save_logs=False, 
    #                                        eval_dataset_name=None, 
    #                                        experiment_name_prefix="evalDiffInit",
    #                                        dataset_selector="diffInit")
    # exit()
    # evaluation_jobs.launch_evaluation_jobs(save_logs=False, 
    #                                        eval_dataset_name=None, 
    #                                        experiment_name_prefix="evalSameInit",
    #                                        dataset_selector="sameInit")
    # exit()
    
    # Launch evaluation jobs for a single dataset
    # Different density datasets : [density68_p_100_diffDen, density70_p_100_diffDen, density72_p_100_diffDen, density74_p_100_diffDen, density76_p_100_diffDen]
    # evaluation_jobs.launch_evaluation_jobs(save_logs=False, 
    #                                        eval_dataset_name="density72_p_100_sameInit_1", 
    #                                        experiment_name_prefix="evalSameInit",
    #                                        dataset_selector="sameInit")
    
    
    # # Evaluate slevel 90 model on slevel 80 dataset
    # evaluation_jobs.launch_evaluation_jobs(save_logs=True, 
    #                                        eval_dataset_name="density72_p_80_uniform", 
    #                                        experiment_name_prefix="evalDiffInitSlevel80",
    #                                        dataset_selector="sameInit")
    
    ############################################################################################################################
    ## Evaluation jobs for generating the train eval matrix (train on one s-level and eval on another s-level)
    # eval all slevel models on slevel 100 dataset
    evaluation_jobs.launch_evaluation_jobs(save_logs=True, 
                                           eval_dataset_name="density72_p_100_uniform", 
                                           experiment_name_prefix="evalDiffInitSlevel100",
                                           dataset_selector="sameInit") # dataset_selector is ignored here since eval_dataset_name is provided
    # eval all slevel models on slevel 95 dataset
    evaluation_jobs.launch_evaluation_jobs(save_logs=True, 
                                           eval_dataset_name="density72_p_95_uniform", 
                                           experiment_name_prefix="evalDiffInitSlevel95",
                                           dataset_selector="sameInit") # dataset_selector is ignored here since eval_dataset_name is provided
    # eval all slevel models on slevel 90 dataset
    evaluation_jobs.launch_evaluation_jobs(save_logs=True, 
                                           eval_dataset_name="density72_p_90_uniform", 
                                           experiment_name_prefix="evalDiffInitSlevel90",
                                           dataset_selector="sameInit") # dataset_selector is ignored here since eval_dataset_name is provided
    # eval all slevel models on slevel 85 dataset
    evaluation_jobs.launch_evaluation_jobs(save_logs=True, 
                                           eval_dataset_name="density72_p_85_uniform", 
                                           experiment_name_prefix="evalDiffInitSlevel85",
                                           dataset_selector="sameInit") # dataset_selector is ignored here since eval_dataset_name is provided
    # eval all slevel models on slevel 80 dataset
    evaluation_jobs.launch_evaluation_jobs(save_logs=True, 
                                           eval_dataset_name="density72_p_80_uniform", 
                                           experiment_name_prefix="evalDiffInitSlevel80",
                                           dataset_selector="sameInit") # dataset_selector is ignored here since eval_dataset_name is provided
    ############################################################################################################################
    
    
    # # Density sweep evaluation 
    # # for density in [64, 66,68, 70, 72, 74, 76]:
    # for density in [76]:
    #     evaluation_jobs.launch_evaluation_jobs(save_logs=False, 
    #                                            eval_dataset_name=f"density{density}_p_100_diffDen", 
    #                                            experiment_name_prefix=f"evalDiffDen{density}",
    #                                            dataset_selector="diffInit")
