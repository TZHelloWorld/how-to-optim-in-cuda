# =====================================================================================================
# 指定基础镜像，安装共用依赖
# =====================================================================================================
FROM micr.cloud.mioffice.cn/cloud-ml/nvidia-pytorch:24.06-py3 as base

# 安装基础软件
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libcap2 \
        libnuma-dev \
        sudo \
        wget \
        vim \
        iputils-ping \
        && \
    apt-get autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

# 换国内源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && mkdir -p /root/.pip/ && \
    echo "[global]" > /root/.pip/pip.conf && \
    echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> /root/.pip/pip.conf

# 编译很慢，github地址为:https://github.com/fanshiqing/grouped_gemm.git
RUN cd /tmp \
    && wget "https://cnbj1-fds.api.xiaomi.net/llm-tz-bucket/code/grouped_gemm@1.1.4.tar" -O grouped_gemm.tar \
    && tar -xvf grouped_gemm.tar \
    && cd grouped_gemm \
    && python setup.py install

# 依赖项
RUN pip install --no-cache-dir sentencepiece \
    && pip install --no-cache-dir wandb \
    && pip install --no-cache-dir flask-restful \
    && pip install --no-cache-dir transformers \
    && pip install --no-cache-dir nltk

# =====================================================================================================
# 给容器添加nccl-test代码，用于后续测试多机网络连通,可删除
# =====================================================================================================
RUN cd /opt \
    && git clone https://github.com/NVIDIA/nccl-tests.git \
    && cd nccl-tests \
    && make

# =====================================================================================================
# todo 登录wandb,将密钥设置为环境变量即可,密钥获取: https://wandb.ai/authorize (先不考虑，可以直接使用crh提供的全局环境变量)
# ====================================================================================================

# =====================================================================================================
# todo 这里考虑将代码复制到容器内，然后考虑执行命令 (ps：暂时先不考虑，使用JuiceFS挂载)
# =====================================================================================================


## 设置默认的工作目录
WORKDIR /workspace

