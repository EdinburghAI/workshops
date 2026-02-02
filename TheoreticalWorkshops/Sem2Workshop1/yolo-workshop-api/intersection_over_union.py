import torch

# uses parallel processing
def get_corners(output_vectors):

    # will be 1d tensors
    center_x = output_vectors[:,1]
    center_y = output_vectors[:,2]
    width = output_vectors[:,3]
    height = output_vectors[:,4]

    x_left = (center_x - width/2).clamp(min=0)
    x_right = (center_x + width/2).clamp(min=0)
    y_bottom = (center_y - height/2).clamp(min=0)
    y_top = (center_y + height/2).clamp(min=0)

    return x_left, x_right, y_bottom, y_top


def iou_batch(predictions, targets):
    # first get the corners of each box
    pred_x_left, pred_x_right, pred_y_bottom, pred_y_top = get_corners(predictions)
    targ_x_left, targ_x_right, targ_y_bottom, targ_y_top = get_corners(targets)

    # now determine corners of the possible intersection box
    inter_x_left = torch.max(pred_x_left, targ_x_left)  # further right left corner
    inter_x_right = torch.min(pred_x_right, targ_x_right)  # further left right corner
    inter_y_bottom = torch.max(pred_y_bottom, targ_y_bottom)  # higher bottom
    inter_y_top = torch.min(pred_y_top, targ_y_top)  # lower top


    # compute intersection aread

    # first check if no overlap
    internal_area_width = (inter_x_right -  inter_x_left).clamp(min=0)
    internal_area_height = (inter_y_top - inter_y_bottom).clamp(min=0)

    internal_area = internal_area_width * internal_area_height

    # now get the union of the two areas
    area_pred_box = predictions[:,3] * predictions[:,4] 
    area_targ_box = targets[:,3] * targets[:,4]

    union_area = area_pred_box + area_targ_box - internal_area

    # add a tiny number which will have no real affect to the ratio but will prevent division by zero
    return internal_area/(union_area+union_area + 1e-7)

