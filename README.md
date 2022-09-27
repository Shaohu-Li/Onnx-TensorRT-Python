# Python 版本实现的简单的 Onnx -> TensorRT

## 1、安装一些必要的安装包
```
pip install --upgrade setuptools pip --user
```
```
pip install nvidia-pyindex
```
```
pip install --upgrade nvidia-tensorrt
```
```
pip install pycuda
```

## 2、具体操作流程
1、下载 yolov7, 并导出 onnx。

2、运行 export_TRT.py, 进行 TRT 模型的导出和图片的测试
具体的步骤参考：examples.ipynb 文件

## demo
在 src 文件下，有一个视频 demo
