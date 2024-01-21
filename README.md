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
1. 检测模型：检测模型的 onnx 按照 utils/sexamples.ipynb 文件中的方式
2. 分割模型：模仿检测先下载分割模型，然后分割模型需要修改一点代码。如果不想修改，data/weight 中存放了导出之后的 onnx 模型，
   ```    
   # 在 seg/export.py 的 164 行 代码下面加入如下代码，再运行 onnx 和 engine 导出的命令即可。
   graph = model_onnx.graph
    # 上面一行为 export onnx 的时候，删除的节点
    # 下面一行为 export engine 的时候，删除的节点
    delete_layer_name = ["onnx::Slice_531", "onnx::Slice_638", "onnx::Slice_744", 
                         "onnx::Slice_536", "onnx::Slice_657", "onnx::Slice_776"]
    for name in delete_layer_name:
        for i in range(len(graph.output)):
            if graph.output[i].name == name:
                graph.output.remove(graph.output[i])
                break 

   ```
   导出命令：
   ```
   python export.py  --weights yolov7-seg.pt --simplify  --include engine --device 0
   ```

2、模型运行
1. 检测运行 detect.py，
   ```
   python detect.py -o data/weight/yolov7x.onnx -e data/weight/yolov7x.trt -ℹ data/imgs/video1.mp4 -m video
   ```
2. 分割运行 segment.py 
   ```
   python segment.py -o data/weight/yolov7-seg.onnx -e data/weight/yolov7-seg.trt -ℹ data/imgs/video1.mp4 -m video
   ```

## demo
在 src 文件下，有成品视频 demo
