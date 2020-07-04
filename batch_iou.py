import torch

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_l = torch.max(box_a[:, 0].unsqueeze(1).expand(A, B), box_b[:, 0].unsqueeze(0).expand(A, B))
    max_r = torch.max(box_a[:, 1].unsqueeze(1).expand(A, B), box_b[:, 1].unsqueeze(0).expand(A, B))
    min_l = torch.min(box_a[:, 0].unsqueeze(1).expand(A, B), box_b[:, 0].unsqueeze(0).expand(A, B))
    min_r = torch.min(box_a[:, 1].unsqueeze(1).expand(A, B), box_b[:, 1].unsqueeze(0).expand(A, B))

    inter = torch.clamp((min_r - max_l), min=0)
    union = max_r - min_l
    iou = inter / union

    return iou


if __name__ == '__main__':
    a = torch.Tensor([[1,3], [1.5,7]])
    b = torch.Tensor([[1,2], [4,5.5]])
    iou = intersect(a, b)

