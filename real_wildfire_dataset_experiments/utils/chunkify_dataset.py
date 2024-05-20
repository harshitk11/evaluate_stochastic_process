import os
import numpy as np

def process_npy_files(dataset_name, src_folder, dst_folder, split):
    src_folder = os.path.join(src_folder, split)
    dst_folder = os.path.join(dst_folder, dataset_name, split+"_chunk_k10_dt50")
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    
    for filename in os.listdir(src_folder):
        if filename.endswith(".npy"):
            # Load the npy file
            file_path = os.path.join(src_folder, filename)
            data = np.load(file_path)
            
            # Ensure the file has enough timesteps to slice
            if data.shape[0] >= 60:
                # Extract the first 60 time steps
                chunk_data = data[:60,:-1,:-1,:]
                print(f"File '{filename}' has {data.shape[0]} timesteps, saving first 60 timesteps.")
                
                # Save the chunk to a new npy file in the destination folder
                chunk_file_path = os.path.join(dst_folder, filename.replace(".npy", "_0_59.npy"))
                np.save(chunk_file_path, chunk_data)
            else:
                print(f"File '{filename}' has less than 60 timesteps, skipping.")
        else:
            print(f"Skipping non-npy file: '{filename}'")

def main():
    dataset_dict = {
            "density72_p_100_sameInit_1": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_same_init/uniform_72_p_0",
            "density72_p_95_sameInit_1": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_same_init/uniform_72_p_95",
            "density72_p_90_sameInit_1": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_same_init/uniform_72_p_9",
            "density72_p_85_sameInit_1": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_same_init/uniform_72_p_85",
            "density72_p_80_sameInit_1": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_same_init/uniform_72_p_8"
            }
    
    dest_dir = "/ssd_2tb/hkumar64/datasets/netlogo_for_baselines/stochasticity_forest_fire_dataset/sameInit_1"
    
    for datasetName, datasetPath in dataset_dict.items():
        process_npy_files(datasetName, datasetPath, dest_dir, "Test")
    
if __name__ == "__main__":
    main()