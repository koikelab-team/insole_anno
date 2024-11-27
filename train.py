import torch
import torch.nn as nn
from dataloader import InsoleDataset
from tqdm import tqdm
import numpy as np
import os
import torch.optim as optim
from models import InsoleAnno, InsoleLSTM, InsoleCNN, InsoleCNNRNN
from tools import denormalize_output, plot_insole_heatmap_gif, log_results
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import wandb

def train(model, train_loader, test_loader, criterion, optimizer, num_epochs=300, device="cuda:0"):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        temp_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=True)
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            # print('x', x, 'y', y)
            optimizer.zero_grad()
            # print('x', x)
            outputs = model(x)
            # print('outputs', outputs)
            t_loss = temporal_consistency_loss(outputs, y)
            mse_loss = criterion(outputs, y)
            loss = t_loss + mse_loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            temp_loss += t_loss.item()
            loop.set_postfix(loss=loss.item())

        log_results(y, outputs, epoch)
        # plot_insole_heatmap_gif(y, outputs, f'./results/output_{epoch}.gif')

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}")
        wandb.log({"train loss": epoch_loss/len(train_loader)})
        wandb.log({"train temporal loss": temp_loss/len(train_loader)})
        test(model, test_loader, criterion, device)

def test(model, dataloader, criterion, device="cuda:0"):
    model.to(device)
    model.eval()
    test_loss = 0
    temp_loss = 0
    for i, data in enumerate(dataloader):
        x, y = data
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        t_loss = temporal_consistency_loss(outputs, y)
        mse_loss = criterion(outputs, y)
        loss = t_loss + mse_loss
        temp_loss += t_loss.item()
        # print(f"Loss: {loss.item()}")
        # if i % 5 == 0: 
        log_results(y, outputs, i, is_test=True)
        plot_insole_heatmap_gif(y, outputs, f'./results/test_{i}.gif')

        # wandb.log({"test_gif": wandb.Video(f'./results/test_{i}.gif', format="gif")})
        test_loss += loss.item()
    wandb.log({"val loss": test_loss/len(dataloader)})
    wandb.log({"val temporal loss": temp_loss/len(dataloader)})
    
def temporal_consistency_loss(output, target, weight=10):
    """
    计算输出与目标值的时间一致性损失。

    Parameters:
    - output: 模型输出, shape (batch_size, seq_len, channels)
    - target: 目标值, shape (batch_size, seq_len, channels)

    Returns:
    - loss: 时间一致性损失，标量
    """
    output_diff = output[:, 1:, :] - output[:, :-1, :]  # 输出的时间差分
    target_diff = target[:, 1:, :] - target[:, :-1, :]  # 目标值的时间差分
    loss = torch.mean((output_diff - target_diff) ** 2)  # L2 范数
    loss = weight * loss
    return loss


if __name__ == "__main__":
    input_dim = 51
    output_dim = 32
    seq_len = 32
    batch_size = 16
    num_epochs = 100
    learning_rate = 1e-4

    data_path = 'dataset'
    dataset = InsoleDataset(data_path, chunk_size=seq_len)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    wandb.init(project="insole-anno")
    model = InsoleAnno(input_dim=input_dim, output_dim=output_dim, seq_len=seq_len)
    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, train_loader, test_loader, criterion, optimizer, num_epochs=num_epochs, device="cuda:0" if torch.cuda.is_available() else "cpu")
    # test(model, test_loader, criterion, device="cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.finish()