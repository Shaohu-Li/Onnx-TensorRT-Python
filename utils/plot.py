import cv2
import torch
import numpy as np
import torch.nn.functional as F

# 前处理
def prepare_img(org_img, img_size, device=torch.device('cuda:0')):
    """对输入的图片进行预处理, 包括 正则化, 不改变宽高比的resize, 还有改变通道顺序

    Args:
        org_img         : 原始的读取的图片

    Returns:
        返回处理好的图片，并返回改变率
    """
    if len(org_img.shape) == 3:
        padded_img = np.ones((img_size[0], img_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(img_size) * 114.0

    img = np.array(org_img)
    ratio = min(img_size[0] / img.shape[0], img_size[1] / img.shape[1])
    new_unpad = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
    resized_img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR,).astype(np.float32)
    
    dw, dh = (int((img_size[1] - new_unpad[0]) / 2), int((img_size[0] - new_unpad[1]) / 2))  # wh padding
    padded_img[dh : new_unpad[1] + dh, dw : new_unpad[0] + dw] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0

    padded_img = padded_img.transpose((2, 0 ,1))
    padded_img = torch.from_numpy(np.ascontiguousarray(padded_img, dtype=np.float32)).to(device)

    return padded_img, ratio

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def crop(masks, boxes):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """

    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def plot_masks(img, masks, colors, alpha=0.5):
    """
    Args:
        img (tensor): img is in cuda, shape: [3, h, w], range: [0, 1]
        masks (tensor): predicted masks on cuda, shape: [n, h, w]
        colors (List[List[Int]]): colors for predicted masks, [[r, g, b] * n]
    Return:
        ndarray: img after draw masks, shape: [h, w, 3]

    transform colors and send img_gpu to cpu for the most time.
    """
    img_gpu = img.clone()
    num_masks = len(masks)
    if num_masks == 0:
        return img.permute(1, 2, 0).contiguous().cpu().numpy() * 255

    # [n, 1, 1, 3]
    # faster this way to transform colors
    colors = torch.tensor(colors, device=img.device).float() / 255.0
    colors = colors[:, None, None, :]
    # [n, h, w, 1]
    masks = masks[:, :, :, None]
    masks_color = masks.repeat(1, 1, 1, 3) * colors * alpha
    inv_alph_masks = masks * (-alpha) + 1
    masks_color_summand = masks_color[0]
    if num_masks > 1:
        inv_alph_cumul = inv_alph_masks[:(num_masks - 1)].cumprod(dim=0)
        masks_color_cumul = masks_color[1:] * inv_alph_cumul
        masks_color_summand += masks_color_cumul.sum(dim=0)

    # print(inv_alph_masks.prod(dim=0).shape) # [h, w, 1]
    img_gpu = img_gpu.flip(dims=[0])  # filp channel for opencv
    img_gpu = img_gpu.permute(1, 2, 0).contiguous()
    # [h, w, 3]
    img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
    return (img_gpu * 255).byte().cpu().numpy()

def scale_masks(img1_shape, masks, img0_shape, ratio_pad=None):
    """
    img1_shape: model input shape, [h, w]
    img0_shape: origin pic shape, [h, w, 3]
    masks: [h, w, num]
    resize for the most time
    """
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    tl_pad = int(pad[1]), int(pad[0])  # y, x
    br_pad = int(img1_shape[0] - pad[1]), int(img1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    # masks_h, masks_w, n
    masks = masks[tl_pad[0]:br_pad[0], tl_pad[1]:br_pad[1]]
    # masks_h, masks_w, n
    masks = cv2.resize(masks, (img0_shape[1], img0_shape[0]))

    # keepdim
    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    return masks

def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Crop before upsample.
    proto_out: [mask_dim, mask_h, mask_w]
    out_masks: [n, mask_dim], n is number of masks after nms
    bboxes: [n, 4], n is number of masks after nms
    shape:input_image_size, (h, w)

    return: h, w, n
    """

    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    return masks.gt_(0.5)

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2    

def plot_box_label(im, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255),line_width=None):
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    lw = line_width or max(round(sum(im.shape) * 0.001), 2)
    cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)

def plot(im, im0, dets, cls_colors, cls_names, proto=None, hide_conf=False, hide_labels=False, show=False):
    # Process predictions
    # for i, det in enumerate(dets):  # per image
    det = dets[0]
    i = 0
    out_img = im0.copy()
    if len(det):
        # Mask plotting ----------------------------------------------------------------------------------------            
        if proto is not None:
            masks       = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[1:], upsample=True)  # HWC
            # Rescale boxes from img_size to im0 size
            det[:, :4]  = scale_coords(im.shape[1:], det[:, :4], im0.shape).round()
            mcolors     = [cls_colors[int(cls)] for cls in det[:, 5]]
            im_masks    = plot_masks(im, masks, mcolors)  # image with masks shape(imh,imw,3)
            out_img     = scale_masks(im.shape[1:], im_masks, im0.shape)  # scale to original h, w
        else:
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[1:], det[:, :4], im0.shape).round()
        # Mask plotting ----------------------------------------------------------------------------------------
        
        # box plotting -----------------------------------------------------------------------------------------
        for *xyxy, conf, cls in reversed(det[:, :6]):
            c = int(cls)  # integer class
            label = None if hide_labels else (cls_names[c] if hide_conf else f'{cls_names[c]} {conf:.2f}')
            plot_box_label(out_img, xyxy, label, color=cls_colors[c])
        # box plotting -----------------------------------------------------------------------------------------
    return out_img