# ComfyUI API 服务

这是一个为 ComfyUI 提供 API 服务的 FastAPI 应用程序。该服务封装了 ComfyUI 的 WebSocket 和 HTTP 接口，提供了更简单的 REST API 接口。

工作流为自制的 style_transfer 工作流，可以参考 workflows/style_transfer.json

## 功能特性

- 图片上传到 ComfyUI 服务器
- 执行 ComfyUI 工作流
- 获取生成历史
- 获取生成的图片
- 支持 WebSocket 实时状态更新
- 自动保存生成的图片

## 环境要求

- Python 3.8+
- ComfyUI 服务器

## 安装步骤

1. 克隆项目到本地

2. 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

1. 确保 ComfyUI 服务器已启动（默认地址：127.0.0.1:8188）

2. 启动 API 服务
```bash
python run.py
```
服务将在 http://0.0.0.0:8000 启动

## 主要依赖

- FastAPI: Web 框架
- aiohttp: 异步 HTTP 客户端
- websockets: WebSocket 客户端
- Pillow: 图像处理
- uvicorn: ASGI 服务器

## 注意事项

- 默认超时时间为 5 分钟
- 生成的图片会自动保存到 output 目录
- 确保 ComfyUI 服务器地址配置正确 
