FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel

# 设置工作目录
WORKDIR /workspace

# 安装 vim、openssh-server、iperf3、iproute2 (ip)、net-tools (ifconfig)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        vim \
        openssh-server \
        iperf3 \
        iproute2 \
        net-tools && \
    mkdir /var/run/sshd && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 复制 requirements.txt 到镜像
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 将本地所有文件复制到镜像中
COPY . .

# 开放端口
EXPOSE 22 16666 5201  

# 默认启动 bash
CMD ["/bin/bash"]
