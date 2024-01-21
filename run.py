
import os
import sys
import cv2
import yaml
import torch
import argparse
from utils.export import init_model
from utils.plot import prepare_img, plot
from utils.box import non_max_suppression

class Model():
    def __init__(self, args) -> None:
        """初始化函数
        Args:
            args: 命令行参数
        """
        self.args = args

        self.engine, self.context, self.bindings, self.binding_addrs =\
            init_model(args.weight, args.half, torch.device(args.device), args.workspace)
        
    def infer_img(self, im0):
        """对单张图片进行推理
        Args:
            img: 输入的图片
        """
        im, _ = prepare_img(im0, self.args.imgsz)

        # 1、对输入的数据进行处理
        self.binding_addrs['images'] = int(im.data_ptr())
        
        # 2、执行推理
        self.context.execute_v2(list(self.binding_addrs.values()))

        # 3、整理输出
        dets = self.bindings['output'].data
        
        proto = None
        nm_proto = 0
        if self.args.mode == "seg":
            nm_proto = 32
            proto = self.bindings['516'].data

        dets = non_max_suppression(dets, self.args.conf_thres, self.args.iou_thres, None, 
                                   self.args.agnostic_nms, max_det=100, nm=nm_proto)

        final_img = plot(im, im0, dets, self.args.colors, self.args.names, proto=proto)
        
        return final_img

    def infer(self):
        """根据包不同的模式，对输入的路径进行推理

        Args:
            img_path: 输入的图片路径
            mode    : 要进行处理的模式. 默认为, "video". choice = ["video", "img"].
        """
        img_path = os.path.realpath(self.args.img_path)
        if self.args.video_mode:
            cap = cv2.VideoCapture(img_path)
            ret, _ = cap.read()
            if not ret:
                print(f"视频读取出错，请检查错误. 当前输入路径为: {img_path}.")
                sys.exit(-1)
            while ret:
                ret, frame = cap.read()
                final_img = self.infer_img(frame)
                cv2.imshow("inference result", final_img)
                if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27 : # 27 对应 Esc
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            if not os.path.isfile(img_path):
                print(f"输入单张图片的路径出错，请检查相应的路径：{img_path}")
                sys.exit(-1)
            frame = cv2.imread(img_path)
            final_img = self.infer_img(frame)
            cv2.imshow("inference result", final_img)
            if cv2.waitKey(-1) == ord('q') or cv2.waitKey(-1) == 27 :
                cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="seg", choices=["det", "seg"], help="The mode of the model, default: segmentation.")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='config.yaml path') # data/coco128.yaml
    parser.add_argument('--weight', type=str, default='data/weights/yolov7-seg.pt', help='Weight needs to load')
    parser.add_argument('--imgsz',  type=int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--workspace', type=int, default=4, help='TensorRT: workspace size (GB)')
    parser.add_argument('--agnostic-nms', action='store_true', help='add agnostic NMS to model')
    parser.add_argument('--topk-per-class', type=int, default=100, help='TF.js NMS: topk per class to keep')
    parser.add_argument('--topk-all', type=int, default=100, help='TF.js NMS: topk for all classes to keep')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='TF.js NMS: IoU threshold')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='TF.js NMS: confidence threshold')    
    parser.add_argument("--video_mode", action='store_true', help="Whether to use video mode. ")
    parser.add_argument("--img_path", default="data/imgs/video1.mp4", help="Input image path. ")
    
    args = parser.parse_args()
    print(args)
    
    with open(args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    args.__dict__.update(config)
    
    Model(args).infer()