# Triton-Inference-Server-tensorrt
此项目展示如何将训练好的模型转换为tensorrt engine并部署到Triton-Inference-Server。

## Install Triton Docker Image
xx.yy为版本号，需要保持一致。
```
$ docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3
$ docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk
$ docker pull nvcr.io/nvidia/tensorrt:<xx.yy>-py3
```
