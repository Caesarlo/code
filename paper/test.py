import plotly.graph_objects as go

# 定义三个矢量（终点坐标）
vectors = [
    (3.1286255054943282, 1.8063127777906796, 7.6230473408312154),
    (-3.1286255054943286, 1.8063127777906796, 7.6230473408312154),
    (0.0000000000000000, -3.6126255555813591, 7.6230473408312154),
]

# 颜色和名称配置
colors = ["red", "green", "blue"]
names = ["Vector 1", "Vector 2", "Vector 3"]

# 创建Figure对象
fig = go.Figure()

# 添加每个矢量的Cone图
for idx, (x, y, z) in enumerate(vectors):
    fig.add_trace(
        go.Cone(
            x=[0],
            y=[0],
            z=[0],  # 起点为原点
            u=[x],
            v=[y],
            w=[z],  # 矢量分量
            anchor="tail",  # 锥体尾部在起点
            sizemode="scaled",  # 根据矢量长度缩放
            sizeref=0.5,  # 调整箭头头大小
            showscale=False,
            colorscale=[[0, colors[idx]], [1, colors[idx]]],  # 单色
            name=names[idx],
        )
    )

# 设置三维场景布局
fig.update_layout(
    title="3D Vector Visualization",
    scene=dict(
        xaxis=dict(title="X", range=[-4, 4]),
        yaxis=dict(title="Y", range=[-4, 4]),
        zaxis=dict(title="Z", range=[0, 8]),
        aspectmode="data",  # 保持坐标轴比例
    ),
    margin=dict(l=0, r=0, b=0, t=30),
)

fig.show()