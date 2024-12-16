# vim: expandtab:ts=4:sw=4
import numpy as np
import cv2
from typing import Optional, List


# def non_max_suppression(boxes, classes, max_bbox_overlap, scores=None):
#     """Suppress overlapping detections.

#     Original code from [1]_ has been adapted to include confidence score.

#     .. [1] http://www.pyimagesearch.com/2015/02/16/
#            faster-non-maximum-suppression-python/

#     Examples
#     --------

#         >>> boxes = [d.roi for d in detections]
#         >>> classes = [d.classes for d in detections]
#         >>> scores = [d.confidence for d in detections]
#         >>> indices = non_max_suppression(boxes, max_bbox_overlap, scores)
#         >>> detections = [detections[i] for i in indices]

#     Parameters
#     ----------
#     boxes : ndarray
#         Array of ROIs (x, y, width, height).
#     max_bbox_overlap : float
#         ROIs that overlap more than this values are suppressed.
#     scores : Optional[array_like]
#         Detector confidence score.

#     Returns
#     -------
#     List[int]
#         Returns indices of detections that have survived non-maxima suppression.

#     """
#     if len(boxes) == 0:
#         return []

#     boxes = boxes.astype(float)
#     pick = []

#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 2] + boxes[:, 0]
#     y2 = boxes[:, 3] + boxes[:, 1]

#     area = (x2 - x1 + 1) * (y2 - y1 + 1)
#     if scores is not None:
#         idxs = np.argsort(scores)
#     else:
#         idxs = np.argsort(y2)

#     while len(idxs) > 0:
#         last = len(idxs) - 1
#         i = idxs[last]
#         pick.append(i)

#         xx1 = np.maximum(x1[i], x1[idxs[:last]])
#         yy1 = np.maximum(y1[i], y1[idxs[:last]])
#         xx2 = np.minimum(x2[i], x2[idxs[:last]])
#         yy2 = np.minimum(y2[i], y2[idxs[:last]])

#         w = np.maximum(0, xx2 - xx1 + 1)
#         h = np.maximum(0, yy2 - yy1 + 1)

#         overlap = (w * h) / area[idxs[:last]]

#         idxs = np.delete(
#             idxs, np.concatenate(
#                 ([last], np.where(overlap > max_bbox_overlap)[0])))

#     return pick

def non_max_suppression(
    boxes: np.ndarray, 
    classes: Optional[np.ndarray], 
    max_bbox_overlap: float, 
    scores: Optional[np.ndarray] = None
) -> List[int]:
    """
    Suppress overlapping detections.

    Parameters
    ----------
    boxes : np.ndarray
        Array of bounding boxes with shape (N, 4).
    classes : Optional[np.ndarray]
        Array of class labels corresponding to each box.
    max_bbox_overlap : float
        IoU threshold for suppression.
    scores : Optional[np.ndarray]
        Confidence scores for each box.

    Returns
    -------
    List[int]
        Indices of bounding boxes that survived NMS.

    """
    # Validate input
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Boxes should be a 2D array with shape (N, 4).")
    if scores is not None and len(scores) != len(boxes):
        raise ValueError("Scores and boxes must have the same length.")
    if classes is not None and len(classes) != len(boxes):
        raise ValueError("Classes and boxes must have the same length.")

    if len(boxes) == 0:
        return []

    # Separate processing for classes if provided
    if classes is not None:
        unique_classes = np.unique(classes)
        final_picks = []
        for cls in unique_classes:
            cls_indices = np.where(classes == cls)[0]
            cls_boxes = boxes[cls_indices]
            cls_scores = scores[cls_indices] if scores is not None else None
            cls_picks = non_max_suppression(cls_boxes, None, max_bbox_overlap, cls_scores)
            final_picks.extend(cls_indices[cls_picks])
        return final_picks

    # Compute coordinates and areas
    boxes = boxes.astype(float)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)

    # Sort indices by scores or bottom-right corner
    idxs = np.argsort(scores) if scores is not None else np.argsort(y2)

    pick = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / (area[idxs[:last]] + area[i] - (w * h))

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > max_bbox_overlap)[0])))

    return pick