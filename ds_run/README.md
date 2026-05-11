# DeepStream 7.0 (V100) 极速重构部署方案

此目录包含了将原有 PyTorch OpenCV 工程改写为 **DeepStream GPU 硬件流水线** 的必要脚手架。包含 Docker 编排与主干 Python 代码。

## 1. 环境准备 (宿主机)

1. 在服务器上安装好包含 NVIDIA Container Toolkit 的 Docker 环境。
   *(确认你的宿主机至少安装了 R535 系列以上的 NVIDIA 显卡驱动)*
2. 你无需在服务器物理机上安装复杂的 CUDA、cuDNN、TensorRT 和 GStreamer 等库。

## 2. Pytorch 模型转为 TensorRT Engine
DeepStream nvinfer 组件不直接认识 `.pt` 文件，必须事先转换：
```bash
# 在含有 ultralytics 的任意环境中执行，或使用官方封装容器
yolo export model=best.pt format=engine workspace=8 half=True
# 将生成的 best.engine 放到本项目与 deepstream_deploy 平级的目录中
```

## 3. 运行极速流水线 (单卡实例为例)
进入 `deepstream_deploy` 目录，通过 docker-compose 直接拉起绑在第一张显卡上的业务容器：

```bash
cd deepstream_deploy
docker-compose up -d deepstream-gpu0
```

### 进入容器二次开发与调试
这个配置已经将你的外层代码目录挂载进了容器。如果你需要修改 `main_ds_yolo.py` 或联调你原有的 `FallVideoBuffer`，只需：

```bash
docker exec -it deepstream_v100_0 bash
cd /opt/nvidia/deepstream/deepstream-7.0/sources/user_project/deepstream_deploy
python3 main_ds_yolo.py
```
> **提示**：如果有报错缺 `easyocr` 或 `ultralytics`，请进入此容器后 `pip install` 即可！因为这已经绑定到宿主机代码目录了，一旦调试跑通，以后重新 `docker run` 就无需大改。

## 4. 扩展到四张 V100 显卡
当你的一张卡承载满了上百路 RTSP 或者遇到算力瓶颈时：
1. 打开 `docker-compose.yml`。
2. 复制 `deepstream-gpu0` 的配置块，将名称改成 `deepstream-gpu1`。
3. 修改显卡设备 ID 绑定：`device_ids: ['1']`。
4. 在启动入口脚本命令里挂载**不同的推流 Camera 配置文件**（例如只负责 camera_1 - camera_30）。
5. `docker-compose up -d deepstream-gpu1`，以此类推。
