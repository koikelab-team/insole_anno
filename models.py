import torch
import torch.nn as nn

class InsoleAnno(nn.Module):
    def __init__(self, input_dim=51, output_dim=32, seq_len=32, num_heads=4, dim_feedforward=256, num_layers=3):
        super(InsoleAnno, self).__init__()

        self.input_proj = nn.Linear(input_dim, output_dim)

        # 定义 Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 自定义注意力层
        # self.attention_layer = nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads, batch_first=True)

        self.output_proj = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        """
        Parameters:
        - x: 输入数据，形状 (batch_size, seq_len, input_dim)

        Returns:
        - 输出数据，形状 (batch_size, seq_len, output_dim)
        - attention_weights: 自注意力权重，形状 (batch_size, num_heads, seq_len, seq_len)
        """
        # 输入投影
        x = self.input_proj(x)  # shape: (batch_size, seq_len, output_dim)

        # Transformer 编码器
        x = self.transformer_encoder(x)

        # 自注意力层
        # attn_output, attn_weights = self.attention_layer(x, x, x)  # 注意力权重: (batch_size, num_heads, seq_len, seq_len)

        # 输出投影
        output = self.output_proj(x)  # shape: (batch_size, seq_len, output_dim)

        return output#, attn_weights
    
class InsoleLSTM(nn.Module):
    def __init__(self, input_dim=51, output_dim=32, seq_len=32, hidden_dim=256, num_layers=3):
        super(InsoleLSTM, self).__init__()
        
        # 输入投影层：将输入维度 input_dim 映射到 LSTM 的隐藏维度 hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # 输出投影层：将 LSTM 的隐藏维度 hidden_dim 映射到输出维度 output_dim
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Parameters:
        - x: 输入数据，形状 (batch_size, seq_len, input_dim)

        Returns:
        - 输出数据，形状 (batch_size, seq_len, output_dim)
        """
        # 投影输入
        x = self.input_proj(x)  # shape: (batch_size, seq_len, hidden_dim)

        # LSTM 编码
        x, _ = self.lstm(x)  # shape: (batch_size, seq_len, hidden_dim)

        # 投影输出
        x = self.output_proj(x)  # shape: (batch_size, seq_len, output_dim)

        return x
    
class InsoleCNN(nn.Module):
    def __init__(self, input_dim=51, output_dim=32, seq_len=32, num_filters=64, kernel_size=3):
        super(InsoleCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2  # 保证输出与输入长度一致
        )
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.conv3 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=output_dim,
            kernel_size=1  # 最后用 1x1 卷积映射到 output_dim
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # 防止过拟合

    def forward(self, x):
        """
        Parameters:
        - x: 输入数据，形状 (batch_size, seq_len, input_dim)

        Returns:
        - 输出数据，形状 (batch_size, seq_len, output_dim)
        """
        # 将输入从 (batch_size, seq_len, input_dim) 转为 (batch_size, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        
        # 卷积层
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        
        # 最后一层映射到 output_dim
        x = self.conv3(x)
        
        # 将输出转回 (batch_size, seq_len, output_dim)
        x = x.permute(0, 2, 1)
        return x
    
class InsoleCNNRNN(nn.Module):
    def __init__(self, input_dim=51, output_dim=32, seq_len=32, num_filters=64, kernel_size=3, hidden_dim=256, num_layers=2):
        super(InsoleCNNRNN, self).__init__()

        # CNN 模块
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        # RNN 模块
        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Parameters:
        - x: 输入数据，形状 (batch_size, seq_len, input_dim)

        Returns:
        - 输出数据，形状 (batch_size, seq_len, output_dim)
        """
        # CNN 提取局部特征
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, input_dim, seq_len)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # 转回 (batch_size, seq_len, num_filters)

        # LSTM 提取时序特征
        x, _ = self.lstm(x)  # 输出形状 (batch_size, seq_len, hidden_dim)

        # 输出映射
        x = self.output_proj(x)  # 输出形状 (batch_size, seq_len, output_dim)

        return x