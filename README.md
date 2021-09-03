# Triton-Inference-Server-tensorrt
此项目展示如何将训练好的模型转换为tensorrt engine并部署到Triton-Inference-Server。

## 下载Docker镜像
获取所需的tritonserver，client，tensorrt镜像，<xx.yy>为版本号，必须保持一致，当前最新版本为21.08。
```
$ docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3
$ docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk
$ docker pull nvcr.io/nvidia/tensorrt:<xx.yy>-py3
```
