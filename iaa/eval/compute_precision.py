import json
from tqdm import tqdm



def ori_bbox(nor_bbox, img_size):
    w = img_size[0]
    h = img_size[1]
    if w>h:
        big = w
        border_w = 0
        border_h = (w-h)//2
    elif w<h:
        big = h
        border_w = (h-w) // 2
        border_h = 0
    elif w==h:
        big = h
        border_w = 0
        border_h = 0
    x1 = nor_bbox[0] * big - border_w
    y1 = nor_bbox[1] * big - border_h
    x2 = nor_bbox[2] * big - border_w
    y2 = nor_bbox[3] * big - border_h
    return [x1, y1, x2, y2]


def compute_iou(bbox1, bbox2):
    """
    computing IoU
    :param bbox1: (x0, y0, x1, y1)
    :param bbox2: (x0, y0, x1, y1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    S_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # computing the sum_area
    sum_area = S_bbox1 + S_bbox2

    # find the each edge of intersect rectangle
    left_line = max(bbox1[0], bbox2[0])
    right_line = min(bbox1[2], bbox2[2])
    top_line = max(bbox1[1], bbox2[1])
    bottom_line = min(bbox1[3], bbox2[3])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect) * 1.0

import sys
gt_info = {}
with open(sys.argv[2], "r") as f:
    for line in tqdm(f):
        info = json.loads(line)
        gt_info[info['sent_id']] = {'bbox': info['bbox'], 'height': info['height'], 'width': info['width']}


filename = sys.argv[1]
modelname = filename.split("/")[5]
testfile = filename.split("/")[6]

with open(sys.argv[1], "r") as f:
    iou_thresh = 0.5
    tp = 0
    fp = 0
    for line in tqdm(f):
        info = json.loads(line)
        idx = info['question_id']
        pred = info['text']
        try:
            gt = gt_info[idx]
            gt_bbox = gt['bbox']
    #         print('gt:',gt_bbox)

            pred_bboxs = pred.split('; ')
            num_bboxs = len(pred_bboxs)

            for i, pred_bbox in enumerate(pred_bboxs):

                pred_bbox = eval(pred_bbox)

                pred_bbox = ori_bbox(pred_bbox, [gt['width'], gt['height']])
                # print('pred:',pred_bbox,'gt:',gt_bbox)

                iou = compute_iou(pred_bbox, gt_bbox)
                if iou >= iou_thresh:
                    tp += 1
                    break
                else:
                    if i == num_bboxs - 1:
                        fp += 1
        except:
            print(pred)
            fp += 1
    precision = tp / (tp + fp)
    print(f'==== REC RESULT: precision = {precision}, tp = {tp}, fp = {fp}')

    with open(sys.argv[3],"a") as fw:
        fw.write(modelname+":  "+testfile+": "+str(precision)+"\n")
