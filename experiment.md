# VisDrone
## faster rcnn 12 epoch
coco metric:\
coco/bbox_mAP: 0.1580  coco/bbox_mAP_50: 0.2820  coco/bbox_mAP_75: 0.1610  coco/bbox_mAP_s: 0.0980  coco/bbox_mAP_m: 0.2400  coco/bbox_mAP_l: 0.2520  data_time: 0.0008  time: 0.0228

## faster rcnn new 12 epoch
我们添加了目标尺度热力图损失，使用的是fpn输出的最后一层，和目标尺度目标尺度热力图做损失，采用的是2维高斯图来生成热力图
### plan 1: 将小、中、大目标生成在不同热力图上
coco metric:\
coco/bbox_mAP: 0.2210  coco/bbox_mAP_50: 0.3660  coco/bbox_mAP_75: 0.2360  coco/bbox_mAP_s: 0.1410  coco/bbox_mAP_m: 0.3290  coco/bbox_mAP_l: 0.3260  data_time: 0.0012  time: 0.0322
### plan 2: 将小、中、大目标生成在同一热力图上