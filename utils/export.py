import os
import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import torch.nn.functional as F
from collections import OrderedDict, namedtuple

def read_trt(engine_path, logger):
    """从已经存在的文件中读取 TRT 模型

    Args:
        engine_path: 已经存在的 TRT 模型的路径

    Returns:
        加载完成的 engine
    """

    engine_path = os.path.realpath(engine_path)
    print(f"From {engine_path} load engine begin . ")
    runtime = trt.Runtime(logger)

    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if (engine is None):
        raise ValueError("加载的 engine 为空，请检查相应的操作. ")
    print(f"From {engine_path} load engine success . ")
    return engine

def onnx_to_trt(file, logger, half, workspace=1, mode='seg'):
    # 构建一个 全局构建器
    builder = trt.Builder(logger)
    builder.max_batch_size = 1

    #---------------------------------#
    # 第一步，读取 onnx
    #---------------------------------#
    # 1-1、设置网络读取的 flag
    network_flags 	= (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    # 1-3、构建一个空的网络计算图, 并设置相应的 flag
    network 		= builder.create_network(network_flags)
    config          = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30
    
    # 1-4、将空的网络计算图和相应的 logger 设置装载进一个 解析器里面
    parser 			= trt.OnnxParser(network, logger)
    # 1-5、打开 onnx 压缩文件，进行模型的解析工作。
    # 解析器 工作完成之后，网络计算图的内容为我们所解析的网络的内容。
    onnx_f 	= os.path.realpath(file)
    if not os.path.isfile(onnx_f):
        print("Onnx file not exist. Please check the onnx file path is right ? ")
        return None
    else:
        if mode == "seg":
            import onnx
            model_onnx = onnx.load(onnx_f)  # load onnx model
            onnx.checker.check_model(model_onnx)  # check onnx model
            graph = model_onnx.graph
            # 下面一行为 export engine 的时候，删除的节点
            delete_layer_name = [ "onnx::Slice_536", "onnx::Slice_657", "onnx::Slice_776"]
            for name in delete_layer_name:
                for i in range(len(graph.output)):
                    if graph.output[i].name == name:
                        graph.output.remove(graph.output[i])
                        break
            onnx.save(model_onnx, onnx_f)

        if not parser.parse_from_file(str(onnx_f)):
            raise RuntimeError(f'failed to load onnx file: {onnx}')
        print("Completed parsing onnx file . ")

    print("Network description:")
    for inp in (network.get_input(i) for i in range(network.num_inputs)):
        # 获取当前转化之前的 输入的 batch_size
        if inp.shape[0] < 1:
            raise ValueError("输入的 batch_size < 0, 请确定输入的参数是否满足要求. ")
        print("Input '{}' with shape {} and dtype {} . ".format(inp.name, inp.shape, inp.dtype))
    for outp in (network.get_output(i) for i in range(network.num_outputs)):
        print("Output '{}' with shape {} and dtype {} . ".format(outp.name, outp.shape, outp.dtype))

    #---------------------------------#
    # 第二步，从构建器 构建引擎, 并 SerializedNetwork 到文件中
    #---------------------------------#
    fp16 = False
    if builder.platform_has_fast_fp16 and half:
        fp16 = True
        config.set_flag(trt.BuilderFlag.FP16)

    engine_f = onnx_f.replace('.onnx', f'.{"fp16" if fp16 else "fp32"}.engine')
    print(f'building FP{16 if fp16 else 32} engine in {engine_f}')
    with builder.build_engine(network, config) as engine, open(engine_f, 'wb') as t:
        t.write(engine.serialize())
    return engine

def init_model(file, half, device, workspace):
    """加载 TRT 模型, 并加载一些多次推理过程共用的参数。
        情况 1、TRT 模型不存在，会先从输入的 onnx 模型创建一个 TRT 模型，并保存，再进行推导；
        情况 2、TRT 模型存在，直接进行推导
    """
    weight_type = file.split('.')[-1]
    # 1、加载 logger 等级
    logger = trt.Logger(trt.Logger.WARNING)

    # 2、加载 TRT 模型
    if weight_type == 'engine':
        engine = read_trt(file, logger)
    elif weight_type == 'onnx':
        engine = onnx_to_trt(file, logger, half, workspace=workspace, mode = 'seg')
    else:
        raise ValueError("输入的模型类型错误，请检查输入的模型类型是否正确. ")

    if (engine is None):
        raise ValueError("加载的 engine 为空，请检查相应的操作. ")
    
    # 3、创建上下管理器，后面进行推导使用
    context = engine.create_execution_context()
    if (context is None):
        raise ValueError("创建的上下文管理器 context 为空，请检查相应的操作. ")

    if device.type == 'cpu':
        device = torch.device('cuda:0')
    # 5、在 cpu 和 gpu 上申请内存
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    bindings = OrderedDict()
    for index in range(engine.num_bindings):
        name = engine.get_binding_name(index)
        dtype = trt.nptype(engine.get_binding_dtype(index))
        shape = tuple(context.get_binding_shape(index))
        im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
        bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        
    return engine, context, bindings, binding_addrs