# 使用 Nvidia DeepStream 6.3 官方镜像以兼顾较老的 V100 (Volta) 架构
FROM nvcr.io/nvidia/deepstream:7.0-triton-multiarch

# 避免安装过程中的交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 设置工作目录
WORKDIR /opt/nvidia/deepstream/deepstream-7.0/sources/user_project

# 1. 替换 apt 镜像源为阿里云
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 2. 安装系统级依赖库 (解决 EasyOCR 或 OpenCV 的依赖)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 3. 配置 pip 源为清华大学镜像并升级 pip
RUN python3 -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 4. 复制依赖文件清单 (你需要创建一个 requirements.txt 放在同级目录)
# COPY requirements.txt /tmp/requirements.txt

# 5. 安装 Python 依赖包 (自动使用上一步配置好的清华源)
# 请根据你的 test.py 实际使用情况修改或在 requirements.txt 中指定
RUN pip3 install \
    ultralytics \
    easyocr \
    opencv-python-headless \
    numpy \
    requests \
    pytz \
    psutil

# 5.5 安装 DeepStream Python 绑定 (pyds) 和 gi (使用本地文件避免网络超时)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-gi python3-gst-1.0 libcairo2-dev default-libmysqlclient-dev libgirepository1.0-dev
RUN pip3 install PyGObject
COPY pyds-1.1.11-py3-none-linux_x86_64.whl /tmp/
RUN pip3 install /tmp/pyds-1.1.11-py3-none-linux_x86_64.whl && rm /tmp/*.whl

# 6. 预构建或绕过 GitHub 下载模型文件
# 你可以提前把 `english_g2.pth` 和 `craft_mlt_25k.pth` 放在跟 Dockerfile 同级的 `models/` 文件夹里
# 这里我们采用在构建时创建 EasyOCR 的本地离线模型目录，等容器跑起来后挂载使用本地模型权重，彻底避开网络请求
RUN mkdir -p /root/.EasyOCR/model
# 在部署时，请确保外部的 model 映射进来，或通过 COPY 提前复制进来
# COPY craft_mlt_25k.pth /root/.EasyOCR/model/
# COPY english_g2.pth /root/.EasyOCR/model/

# 设置启动命令，默认进入bash，或者直接启动深流程序
CMD ["bash"]
