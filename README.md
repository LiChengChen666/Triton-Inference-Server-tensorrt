# Triton-Inference-Server-tensorrt
此项目展示如何将训练好的模型转换为tensorrt engine并部署到Triton-Inference-Server。

## 下载Docker镜像
获取所需的server，client，tensorrt镜像，<xx.yy>为版本号，必须保持一致，当前最新版本为21.08。
```
$ docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3
$ docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk
$ docker pull nvcr.io/nvidia/tensorrt:<xx.yy>-py3
```
## 构建TensorRT engine
参考wang-xinyu/tensorrtx项目，生成TensorRT engine。
```
git clone https://github.com/wang-xinyu/tensorrtx.git
docker run --gpus all -it --rm -v$(pwd)/tensorrtx:/tensorrtx nvcr.io/nvidia/tensorrt:<xx.yy>-py3
```
进入容器后，参照tensorrtx的readme.md生成对应的`engine`以及`libmyplugin.so`（建议先安装opencv）。
```
apt-get install libopencv-dev
```
## 部署到Triton Inference Server
```
mkdir -p model_repository/<你模型的名字>/1
mkdir model_repository/plugins
#把生成的engine和libmyplugin.so拷贝到储存库下面。
cp <名字>.engine model_repository/<你模型的名字>/1/model.plan
cp libmyplugin.so model_repository/plugins
```
然后运行tritonserver
```
docker run --gpus all --rm --shm-size=1g --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v$(pwd)/model_repository:/models -v$(pwd)/model_repository/plugins:/plugins --env LD_PRELOAD=/plugins/libmyplugin.so nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/models --strict-model-config=false --grpc-infer-allocation-pool-size=16 --log-verbose 1
```
