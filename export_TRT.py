
import argparse
from ast import arg, parse
from genericpath import isfile
import os
import sys
import cv2
import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TRT():
    def __init__(self, onnx_file_path, engine_file_path, precision_flop, end2end, conf_thres, nms_thres, score_thres) -> None:
        self.onnx_file_path     = onnx_file_path
        self.engine_file_path   = engine_file_path
        self.precision_flop     = precision_flop
        self.end2end            = end2end

        self.inputs             = []
        self.outputs            = []
        self.bindings           = []

        self.img_size           = (640, 640)
        self.conf_threshold     = conf_thres
        self.nms_threshold      = nms_thres
        self.score_threshold    = score_thres

        self.COCO ={
                    "label":[ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
                        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
                        'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
                        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush' ],

                    "color":[[244, 67, 54], [233, 30, 99], [156, 39, 176], [103, 58, 183], [100, 30, 60], [63, 81, 181], [33, 150, 243], [3, 169, 244], [0, 188, 212],[20, 55, 200],
                            [0, 150, 136], [76, 175, 80], [139, 195, 74], [205, 220, 57], [70, 25, 100], [255, 235, 59], [255, 193, 7], [255, 152, 0], [255, 87, 34], [90, 155, 50],
                            [121, 85, 72], [158, 158, 158], [96, 125, 139], [15, 67, 34], [98, 55, 20], [21, 82, 172], [58, 128, 255], [196, 125, 39], [75, 27, 134], [90, 125, 120],
                            [121, 82, 7], [158, 58, 8], [96, 25, 9], [115, 7, 234], [8, 155, 220], [221, 25, 72], [188, 58, 158], [56, 175, 19], [215, 67, 64], [198, 75, 20],
                            [62, 185, 22], [108, 70, 58], [160, 225, 39], [95, 60, 144], [78, 155, 120], [101, 25, 142], [48, 198, 28], [96, 225, 200], [150, 167, 134], [18, 185, 90],
                            [21, 145, 172], [98, 68, 78], [196, 105, 19], [215, 67, 84], [130, 115, 170], [255, 0, 255], [255, 255, 0], [196, 185, 10], [95, 167, 234],[18, 25, 190],
                            [0, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [155, 0, 0], [0, 155, 0], [0, 0, 155], [46, 22, 130], [255, 0, 155], [155, 0, 255], 
                            [255, 155, 0],[155, 255, 0], [0, 155, 255], [0, 255, 155], [18, 5, 40], [120, 120, 255], [255, 58, 30], [60, 45, 60], [75, 27, 244], [128, 25, 70]]
                    }

        self.Init_model()

    def Init_model(self):
        """加载 TRT 模型, 并加载一些多次推理过程共用的参数。
            情况 1、TRT 模型不存在，会先从输入的 onnx 模型创建一个 TRT 模型，并保存，再进行推导；
            情况 2、TRT 模型存在，直接进行推导
        """
        # 1、加载 logger 等级
        self.logger = trt.Logger(trt.Logger.WARNING)

        # 2、加载 TRT 模型
        if os.path.isfile(self.engine_file_path):
            self.engine = self.readTrtFile(self.engine_file_path)
            assert self.engine, "从 TRT 文件中读取的 engine 为 None ! "
        else:
            self.engine = self.onnxToTRTModel(self.onnx_file_path, self.engine_file_path, self.precision_flop)
            assert self.engine, "从 onnx 文件中转换的 engine 为 None ! "
        
        # 3、创建上下管理器，后面进行推导使用
        self.context = self.engine.create_execution_context()
        assert self.context, "创建的上下文管理器 context 为空，请检查相应的操作"

        # 4、创建数据传输流，在 cpu <--> gpu 之间传输数据的时候使用。
        self.stream = cuda.Stream()

        # 5、在 cpu 和 gpu 上申请内存
        for binding in self.engine:
            # 对应的输入输出内容的 个数，！！！注意是个数，不是内存的大小，
            size = trt.volume(self.engine.get_binding_shape(binding))
            # 内存的类型，如 int， bool。单个数据所占据的内存大小
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # 个数 * 单个内存的大小 = 内存的真实大小，先申请 cpu 上的内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            # 分配 gpu 上的内存
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            print("size: {}, dtype: {}, device_mem: {}".format(size, dtype, device_mem))
            # 区分输入的和输出 申请的内存
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def inference(self, img_path, mode="video"):
        """根据包不同的模式，对输入的路径进行推理

        Args:
            img_path: 输入的图片路径
            mode    : 要进行处理的模式. 默认为, "video". choice = ["video", "img"].
        """
        img_path = os.path.realpath(img_path)

        if mode == "video":
            cap = cv2.VideoCapture(img_path)
            ret, frame = cap.read()
            if not ret:
                print("视频读取出错，请检查错误. 当前输入路径为: {}. ".format(img_path))
                sys.exit(-1)
            while ret:
                ret, frame = cap.read()
                img, ratio = self.prepareImage(frame)
                engine_infer_output = self.infer_single_img(img)
                final_img = self.post_process(engine_infer_output, frame, ratio)
                cv2.imshow("TRT inference result", final_img)
                if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27 : # 27 对应 Esc
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            if not os.path.isfile(img_path):
                print("输入单张图片的路径出错，请检查相应的路径：{}".format(img_path))
                sys.exit(-1)
            frame = cv2.imread(img_path)
            img, ratio = self.prepareImage(frame)
            engine_infer_output = self.infer_single_img(img)
            final_img = self.post_process(engine_infer_output, frame, ratio)
            cv2.imshow("TRT inference result", final_img)
            if cv2.waitKey(-1) == ord('q') or cv2.waitKey(-1) == 27 :
                cv2.destroyAllWindows()

    def infer_single_img(self, img):
        """对单张图片进行推理

        Args:
            img: 输入的图片

        Returns:
            返回 trt 推理的结果
        """

        # 1、对输入的数据进行处理
        self.inputs[0]['host'] = np.ravel(img) # 目前数据是放在 cpu 上
        # 2、将输入的数据同步到 gpu 上面 , 从 host -> device
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

        # 3、执行推理（Execute / Executev2）
        # execute_async_v2  ： 对批处理异步执行推理。此方法仅适用于从没有隐式批处理维度的网络构建的执行上下文。
        # execute_v2：      ： 在批次上同步执行推理。此方法仅适用于从没有隐式批处理维度的网络构建的执行上下文。
        # 同步和异步的差异    ： 在同一个上下文管理器中，程序的执行是否严格按照从上到下的过程。
        #                     如，连续输入多张图片，同步 会等处理完结果再去获得下一张，异步会开启多线程，提前处理数据 
        self.context.execute_async_v2(
                                bindings=self.bindings, # 要进行推理的数据，放进去的时候，只有输入，出来输入、输出都有了
                                stream_handle=self.stream.handle # 将在其上执行推理内核的 CUDA 流的句柄。
                )
        # 4、Buffer 拷贝操作	Device to Host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        # 5、将 stream 中的数据进行梳理
        self.stream.synchronize()

        # 6、整理输出
        engine_infer_output = []
        for out in self.outputs:
            out['host'] = np.reshape(out['host'], (-1, 85))
            engine_infer_output.append(out['host'])
        engine_infer_output = np.concatenate(engine_infer_output, 0) 

        return engine_infer_output

    def result_visual(self, img, boxes, scores, cls_ids, classes_and_colors):
        """对得到的结果进行可视化

        Args:
            img                 : 原始输入的图片
            boxes               : 最终的检测框
            scores              : 最终检测框的得分
            cls_ids             : 最终检测框对应的类别序号
            classes_and_colors  : coco数据集类别和颜色

        Returns:
            _description_
        """

        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < self.conf_threshold:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            color = (classes_and_colors["color"][cls_id])
            text = '{}:{:.1f}%'.format(classes_and_colors["label"][cls_id], score * 100)
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.6, 2)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            cv2.rectangle(img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])), color, 1)
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.6, color, thickness=2)
            
        return img

    # 前处理
    def prepareImage(self, org_img):
        """对输入的图片进行预处理, 包括 正则化, 不改变宽高比的resize, 还有改变通道顺序

        Args:
            org_img         : 原始的读取的图片

        Returns:
            返回处理好的图片，并返回改变率
        """
        netinput_size = self.img_size
        if len(org_img.shape) == 3:
            padded_img = np.ones((netinput_size[0], netinput_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(netinput_size) * 114.0

        img = np.array(org_img)
        ratio = min(netinput_size[0] / img.shape[0], netinput_size[1] / img.shape[1])

        resized_img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)), interpolation=cv2.INTER_LINEAR,).astype(np.float32)

        padded_img[: int(img.shape[0] * ratio), : int(img.shape[1] * ratio)] = resized_img

        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0

        padded_img = padded_img.transpose((2, 0 ,1))
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

        return padded_img, ratio

    # 如果在非 end2end 的情况下的时候, 我们需要对当输出的结果进行 NMS
    def post_process(self, engine_infer_output, origin_img, ratio):
        """对网络输出的结果进行后处理

        Args:
            engine_infer_output : 网络输出的结果，-> ( 25200, 85)
            origin_img          : 送入网络之前的原始图片
            ratio               : 原始图片的大小 / 送入网络的图片大小

        Returns:
            最终绘制完层检测框的图片
        """
        # 再没有进行非极大值抑制的情况下，原始网络输出为 25200*85 = （ 20 * 20 + 40 *40 + 80 * 80） * 85 * 3 (三个输出头)
        if self.end2end :
            num, final_boxes, final_scores, final_cls_inds = engine_infer_output
            final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
            dets = np.concatenate(
                                [final_boxes[:num[0]], 
                                np.array(final_scores)[:num[0]].reshape(-1, 1), 
                                np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], 
                                axis=-1
                                )
        else:
            dets = self.non_max_suppression(engine_infer_output, ratio)

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:,:4], dets[:, 4], dets[:, 5]

            origin_img = self.result_visual(origin_img, final_boxes, final_scores, final_cls_inds, self.COCO)
        return origin_img

    def non_max_suppression(self, prediction, ratio, num_classes = 80):
        """对检测头输出的多个检测框, 进行非极大值抑制

        Args:
            prediction  : 检测头输出的全部的检测框，具体的维度信息为 (25200, 85)
            ratio       : 输入图片被缩放的系数
        Returns:
            输出每张图片上进行过非极大值抑制的结果，最终的维度为：(n, 6); 6 -> [xyxy, conf, cls]
        """
        # 0、最终的输出结果
        boxes_after_nms = []

        # 1、先去除一些置信度比较低的
        mask = np.squeeze(prediction[..., 4:5] > self.score_threshold)
        prediction = prediction[mask]

        # 2、得到每个检测框的的得分数，-> box_scores = obj_conf * cls_conf
        scores               = prediction[:, 4:5] * prediction[:, 5:]
        # 3、转换 (center x, center y, width, height) to (x1, y1, x2, y2), 并转换为适应图片的大小
        boxes                = self.xywh2xyxy(prediction[:, :4]) / ratio 

        # 4、按照不同的 类别 进行 nms
        for class_i in range(num_classes):
            cls_scores = scores[:, class_i]
            cls_score_mask = cls_scores > self.score_threshold
            if cls_score_mask.sum() == 0:
                continue
            else:
                cls_scores = cls_scores[cls_score_mask]
                cls_boxes = boxes[cls_score_mask]

                keep = self.nms(cls_boxes, cls_scores, 0.45)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * class_i
                    dets = np.concatenate([cls_boxes[keep], cls_scores[keep, None], cls_inds], 1)
                    boxes_after_nms.append(dets)
        if len(boxes_after_nms) == 0:
            return None
        return np.concatenate(boxes_after_nms, 0)

    def nms(self,boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def box_iou(self, boxes_preds, boxes_labels, box_format="midpoint"):
        """

        计算两个框之间的面积的交并比(iou)的函数

        Parameters:
            boxes_preds (tensor) : 网络预测出来的框的坐标 (BATCH_SIZE, 4)
            boxes_labels (tensor): 真实标签下的框的坐标   (BATCH_SIZE, 4)
            box_format (str)     : 选择自己的模式, midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

        Returns:
            tensor: 返回检测框之间的 iou
        """

        if box_format == "midpoint":
            box1_x1 = boxes_preds[..., 0:1]  - boxes_preds[..., 2:3] / 2
            box1_y1 = boxes_preds[..., 1:2]  - boxes_preds[..., 3:4] / 2
            box1_x2 = boxes_preds[..., 0:1]  + boxes_preds[..., 2:3] / 2
            box1_y2 = boxes_preds[..., 1:2]  + boxes_preds[..., 3:4] / 2
            box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
            box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
            box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
            box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

        if box_format == "corners":
            box1_x1 = boxes_preds[..., 0:1]
            box1_y1 = boxes_preds[..., 1:2]
            box1_x2 = boxes_preds[..., 2:3]
            box1_y2 = boxes_preds[..., 3:4]
            box2_x1 = boxes_labels[..., 0:1]
            box2_y1 = boxes_labels[..., 1:2]
            box2_x2 = boxes_labels[..., 2:3]
            box2_y2 = boxes_labels[..., 3:4]

        x1 = max(box1_x1, box2_x1)
        y1 = max(box1_y1, box2_y1)
        x2 = min(box1_x2, box2_x2)
        y2 = min(box1_y2, box2_y2)

        # 确保交集的框的宽高不会是负数
        intersection = max(0, (x2 - x1)) * max(0, (y2 - y1))
        box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
        box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

        return intersection / (box1_area + box2_area - intersection + 1e-6)

    def readTrtFile(self, engine_file_path):
        """从已经存在的文件中读取 TRT 模型

        Args:
            engine_file_path: 已经存在的 TRT 模型的路径

        Returns:
            加载完成的 engine
        """

        engine_file_path = os.path.realpath(engine_file_path)
        print("Loading TRT fil from : {}".format(engine_file_path))

        runtime = trt.Runtime(self.logger)

        with open(engine_file_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        assert engine, "反序列化之后的 engien 为空，确保转换过程的正确性 . "
        print("From {} load engine sucess . ".format(engine_file_path))
        return engine

    def onnxToTRTModel(self, onnx_file_path, engine_file_path, precision_flop):
        """构建期 -> 转换网络模型为 TRT 模型

        Args:
            onnx_file_path  : 要转换的 onnx 模型的路径
            engine_file_path: 转换之后的 TRT engine 的路径
            precision_flop  : 转换过程中所使用的精度

        Returns:
            转化成功: engine
            转换失败: None
        """
        #---------------------------------#
        # 准备全局信息
        #---------------------------------#
        # 构建一个 构建器
        builder = trt.Builder(self.logger)
        builder.max_batch_size = 1

        #---------------------------------#
        # 第一步，读取 onnx
        #---------------------------------#
        # 1-1、设置网络读取的 flag
        # EXPLICIT_BATCH 相教于 IMPLICIT_BATCH 模式，会显示的将 batch 的维度包含在张量维度当中，
        # 有了 batch大小的，我们就可以进行一些必须包含 batch 大小的操作了，如 Layer Normalization。  
        #不然在推理阶段，应当指定推理的 batch 的大小。目前主流的使用的 EXPLICIT_BATCH 模式
        network_flags 	= (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        # 1-3、构建一个空的网络计算图
        network 		= builder.create_network(network_flags)
        # 1-4、将空的网络计算图和相应的 logger 设置装载进一个 解析器里面
        parser 			= trt.OnnxParser(network, self.logger)
        # 1-5、打开 onnx 压缩文件，进行模型的解析工作。
        # 解析器 工作完成之后，网络计算图的内容为我们所解析的网络的内容。
        onnx_file_path 	= os.path.realpath(onnx_file_path) # 将路径转换为绝对路径防止出错
        if not os.path.isfile(onnx_file_path):
            print("ONNX file not exist. Please check the onnx file path is right ? ")
            return None
        else:
            with open(onnx_file_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the onnx file {} . ".format(onnx_file_path))
                    # 出错了，将相关错误的地方打印出来，进行可视化处理`-`
                    for error in range(parser.num_errors):
                        print(parser.num_errors)
                        print(parser.get_error(error))
                    return None
            print("Completed parsing ONNX file . ")
        # 6、将转换之后的模型的输入输出的对应的大小进行打印，从而进行验证
        for i in range(network.num_outputs):
            print(i)
        inputs 	= [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]

        print("Network Description")
        batch_size = 0
        for inp in inputs:
            # 获取当前转化之前的 输入的 batch_size
            batch_size = inp.shape[0]
            print("Input '{}' with shape {} and dtype {} . ".format(inp.name, inp.shape, inp.dtype))
        for outp in outputs:
            print("Output '{}' with shape {} and dtype {} . ".format(outp.name, outp.shape, outp.dtype))
        # 确保 输入的 batch_size 不为零
        assert batch_size > 0, "输入的 batch_size < 0, 请确定输入的参数是否满足要求. "

        #---------------------------------#
        # 第二步，转换为 TRT 模型
        #---------------------------------#
        # 2-1、设置 构建器 的 相关配置器
        # 应当丢弃老版本的 builder. 进行设置的操作
        config = builder.create_builder_config()
        # 2-2、设置 可以为 TensorRT 提供策略的策略源。如CUBLAS、CUDNN 等
        # 也就是在矩阵计算和内存拷贝的过程中选择不同的策略
        # config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS))
        # 2-3、给出模型中任一层能使用的内存上限，这里是 2^30,为 2GB
        # 每一层需要多少内存系统分配多少，并不是每次都分 2 GB
        config.max_workspace_size = 1 << 30
        # 2-4、设置 模型的转化精度
        config.set_flag(trt.BuilderFlag.FP16)

        # 2-5，从构建器 构建引擎
        engine = builder.build_engine(network, config)

        #---------------------------------#
        # 第三步，生成 SerializedNetwork
        #---------------------------------#
        # 3-1、删除已经已经存在的版本
        engine_file_path 	= os.path.realpath(engine_file_path) # 将路径转换为绝对路径防止出错
        if os.path.isfile(engine_file_path):
            try:
                os.remove(engine_file_path)
            except Exception:
                print("Cannot removing existing file: {} ".format(engine_file_path))

        print("Creating Tensorrt Engine: {}".format(engine_file_path))

        # 3-2、打开要写入的 TRT engine，利用引擎写入
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print("ONNX -> TRT Success。 Serialized Engine Saved at: {} . ".format(engine_file_path))

        return engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", help="Input onnx model path. ")
    parser.add_argument("-e", "--engine", help="Output TRT model path. ")
    parser.add_argument("-p", "--precision", default="fp16", choices=["fp32", "fp16", "int8"], 
                        help="The precision mode to build in, either 'fp32', 'fp16' or 'int8', default: 'fp16'")
    parser.add_argument("--end2end", default=False, action="store_true",
                        help="export the engine include nms plugin, default: False")
    parser.add_argument("--conf_thres", default=0.4, type=float,
                        help="The conf threshold for the nms, default: 0.5")
    parser.add_argument("--iou_thres", default=0.5, type=float,
                        help="The iou threshold for the nms, default: 0.45")
    parser.add_argument("--scores_thres", default=0.25, type=float,
                        help="The scores threshold for the nms, default: 0.25")
    parser.add_argument("-i", "--img_path", default="python/src/video1.mp4")
    parser.add_argument("-m", "--mode", default="video")
    
    args = parser.parse_args()
    print(args)
    if not all([args.onnx, args.engine]):
        parser.print_help()
        print("These arguments are required: --onnx and --engine")
        sys.exit(1)
    trt_model = TRT(args.onnx, args.engine, args.precision, args.end2end, args.conf_thres, args.iou_thres, args.scores_thres)
    if args.img_path:
        trt_model.inference(args.img_path, mode=args.mode)