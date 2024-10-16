# FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:2.0.0-py3.9.12-cuda11.8.0-u22.04
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.12-py3.9.12-cuda11.3.1-u20.04


# 如有安装其他软件的需求
RUN pip3 install numpy pandas --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# 复制代码到镜像仓库
COPY app /app

# 指定工作目录
WORKDIR /app

# 容器启动运行命令
CMD ["bash", "run.sh"]