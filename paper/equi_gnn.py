import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, DataLoader

class EGNNLayer(MessagePassing):
    """等变图神经网络层"""
    def __init__(self, node_dim, coord_dim, edge_dim):
        super().__init__(aggr='mean')
        self.node_dim = node_dim
        self.coord_dim = coord_dim
        self.edge_dim = edge_dim
        
        # 消息网络
        self.msg_mlp = nn.Sequential(
            nn.Linear(2*node_dim + edge_dim + 1, 2*node_dim),
            nn.SiLU(),
            nn.Linear(2*node_dim, node_dim)
        )
        
        # 坐标更新网络
        self.coord_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 1)
        )

    def forward(self, x, pos, edge_index, edge_attr):
        # 保存原始坐标用于消息传递
        pos = pos.detach().requires_grad_(True)
        return self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)

    def message(self, x_i, x_j, pos_i, pos_j, edge_attr):
        # 计算相对坐标和距离
        rel_coord = pos_j - pos_i
        distance = torch.norm(rel_coord, dim=-1, keepdim=True)
        
        # 构建消息输入
        message_input = torch.cat([x_i, x_j, edge_attr, distance], dim=-1)
        message = self.msg_mlp(message_input)
        return message

    def update(self, aggr_msg, x, pos):
        # 更新节点特征
        new_x = x + aggr_msg
        
        # 更新坐标
        coord_update = self.coord_mlp(aggr_msg)
        new_pos = pos + coord_update * torch.randn_like(pos)  # 保持等变性
        
        return new_x, new_pos

class EGNN(nn.Module):
    """等变图神经网络模型"""
    def __init__(self, node_dim=64, coord_dim=3, edge_dim=4, num_layers=4):
        super().__init__()
        self.embedding = nn.Linear(node_dim, node_dim)
        self.layers = nn.ModuleList([
            EGNNLayer(node_dim, coord_dim, edge_dim) 
            for _ in range(num_layers)
        ])
        self.output_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim//2),
            nn.SiLU(),
            nn.Linear(node_dim//2, 1)
        )

    def forward(self, data):
        x = self.embedding(data.x)
        pos = data.pos
        
        for layer in self.layers:
            x, pos = layer(x, pos, data.edge_index, data.edge_attr)
        
        # 全局池化
        graph_embedding = torch.mean(x, dim=0)
        return self.output_mlp(graph_embedding)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = torch.nn.L1Loss()(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f'Epoch: {epoch}, Train MAE: {total_loss/len(train_loader):.4f}')

def test(model, device, test_loader):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            total_loss += torch.nn.L1Loss()(out, batch.y).item()
    
    print(f'Test MAE: {total_loss/len(test_loader):.4f}')

if __name__ == "__main__":
    # 示例使用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模拟数据集（实际使用时需要替换为真实数据）
    dataset = [
        Data(
            x=torch.randn(5, 64),  # 节点特征
            pos=torch.randn(5, 3), # 3D坐标
            edge_index=torch.tensor([[0,1,2,3,4], [1,2,3,4,0]]), 
            edge_attr=torch.randn(5, 4), 
            y=torch.randn(1)
        ) for _ in range(100)
    ]
    
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    model = EGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(1, 101):
        train(model, device, loader, optimizer, epoch)
        test(model, device, loader) 