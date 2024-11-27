import torch
import torch.nn as nn
import torch.nn.functional as F 
import csv
import torch.utils.data as data
import os
import pandas as pd
import numpy as np
import ast
import re



class InsoleDataset(data.Dataset):
    def __init__(self, data_path, chunk_size=32, transform=None):
        self.data_path = data_path
        self.pose_list = []
        self.insole_list = []
        self.transform = transform

        leftp_idx = [i for i in range(1,17)]
        rightp_idx = [i for i in range(26,42)]

        def parse_line(line):
            # 使用正则表达式提取括号内的数字组
            matches = re.findall(r'\[([^\]]+)\]', line)
            parsed_data = [np.fromstring(match, sep=' ') for match in matches]
            return np.array(parsed_data)
        
        for file in os.listdir(data_path+'/downsampled_pose'):
            if file.endswith(".csv") and file in os.listdir(data_path+'/insole'):
                pose_path = os.path.join(data_path+'/downsampled_pose', file)
                insole_path = os.path.join(data_path+'/insole', file)
                pose_df = pd.read_csv(pose_path, header=None)
                with open(pose_path, 'r') as f:
                    pose_lines = f.readlines()
                pose_df = [parse_line(line.strip()) for line in pose_lines]
                
                insole_df = pd.read_csv(insole_path, header=None, dtype=np.float32)
                insole_data = pd.concat([insole_df.iloc[:, leftp_idx], insole_df.iloc[:, rightp_idx]], axis=1)
                if len(pose_df) > len(insole_data):
                    pose_df = pose_df[:len(insole_data)]
                elif len(pose_df) < len(insole_data):
                    insole_data = insole_data[:len(pose_df)]

                pose_chunk = self.chunk_data(pose_df, chunk_size)
                insole_chunk = self.chunk_data(insole_data.values, chunk_size)
                self.pose_list.extend(pose_chunk)
                self.insole_list.extend(insole_chunk)

        print('pose_list:', len(self.pose_list), 'insole_list:', len(self.insole_list))
        self.normalize_data()


    def normalize_data(self):
        # Flatten the lists for calculating min and max
        pose_flat = np.vstack(self.pose_list)
        insole_flat = np.vstack(self.insole_list)
        insole_flat = np.nan_to_num(insole_flat)
        # print(insole_flat.shape)
        
        # Calculate min and max
        self.pose_min, self.pose_max = pose_flat.min(axis=0), pose_flat.max(axis=0)
        self.insole_min, self.insole_max = insole_flat.min(axis=0), insole_flat.max(axis=0)
        # print(self.pose_min, self.pose_max, self.insole_min, self.insole_max)
        save_normalization_params(self.pose_min, self.pose_max, self.insole_min, self.insole_max)

        # Normalize pose_list
        self.pose_list = [
            2 * (chunk - self.pose_min) / (self.pose_max - self.pose_min) - 1 
            for chunk in self.pose_list
        ]
        
        # Normalize insole_list
        self.insole_list = [
            2 * (chunk - self.insole_min) / (self.insole_max+(1e-4) - self.insole_min) - 1 
            for chunk in self.insole_list
        ]
        self.insole_list = [np.nan_to_num(chunk, nan=0.0, posinf=1e6, neginf=-1e6) for chunk in self.insole_list]

    def chunk_data(self, data, chunk_size):
        return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size) if i+chunk_size <= len(data)]
        
        
    def __len__(self):
        return len( self.pose_list)
    
    def __getitem__(self, idx):
        pose = self.pose_list[idx]
        insole = self.insole_list[idx]
        if self.transform:
            data = self.transform(data)

        pose = torch.tensor(pose).float()
        pose = pose.reshape(pose.shape[0], -1)
        insole = torch.tensor(insole).float()
        return pose, insole
    
# Save min and max values to .npy files
def save_normalization_params(pose_min, pose_max, insole_min, insole_max, save_dir="./dataset"):
    """
    Save normalization parameters (min and max values) for pose and insole data.

    Parameters:
    - pose_min, pose_max: Min and max values for pose data (arrays).
    - insole_min, insole_max: Min and max values for insole data (arrays).
    - save_dir: Directory where the .npy files will be saved.
    """
    np.save(f"{save_dir}/pose_min.npy", pose_min)
    np.save(f"{save_dir}/pose_max.npy", pose_max)
    np.save(f"{save_dir}/insole_min.npy", insole_min)
    np.save(f"{save_dir}/insole_max.npy", insole_max)
    print(f"Normalization parameters saved to {save_dir}")



if __name__=='__main__':
    data_path = 'dataset'
    dataset = InsoleDataset(data_path)
    print(len(dataset))
    print(dataset[0][0], dataset[0][1])