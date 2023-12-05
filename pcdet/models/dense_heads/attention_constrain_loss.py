# Copyright (c) Gorilla-Lab. All rights reserved.
from typing import Sequence
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import stack as tstack

def rotation_2d(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.

    Returns:
        float array: same shape as points
    """
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    rot_mat_T = torch.stack([tstack([rot_cos, -rot_sin]), tstack([rot_sin, rot_cos])])
    return torch.einsum("aij,jka->aik", (points, rot_mat_T))


def torch_to_np_dtype(ttype):
    type_map = {
        torch.float16: np.dtype(np.float16),
        torch.float32: np.dtype(np.float32),
        torch.float16: np.dtype(np.float64),
        torch.int32: np.dtype(np.int32),
        torch.int64: np.dtype(np.int64),
        torch.uint8: np.dtype(np.uint8),
    }
    return type_map[ttype]

def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
        dtype (output dtype, optional): Defaults to np.float32

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    dtype = torch_to_np_dtype(dims.dtype)
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1
    ).astype(dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start from minimum point
    # for 3d boxes, please draw them by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dtype)
    corners_norm = torch.from_numpy(corners_norm).type_as(dims)
    corners = dims.view(-1, 1, ndim) * corners_norm.view(1, 2 ** ndim, ndim)
    return corners

def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.

    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.view(-1, 1, 2)
    return corners

def points_in_convex_polygon_torch(points, polygon, clockwise=True):
    """check points is in convex polygons. may run 2x faster when write in
    cython(don't need to calculate all cross-product between edge and point)
    Args:
        points: [num_points, 2] array.
        polygon: [num_polygon, num_points_of_polygon, 2] array.
        clockwise: bool. indicate polygon is clockwise.
    Returns:
        [num_points, num_polygon] bool array.
    """
    # first convert polygon to directed lines
    num_lines = polygon.shape[1]
    #looks like pytorch1.3 hates irregular indexing
    #polygon_next = polygon[:, [num_lines - 1] + list(range(num_lines - 1)), :]
    try:
        polygon_next = torch.cat([polygon[:, -1:, :], polygon[:, :-1, :]], dim=1)
    except:
        import pdb; pdb.set_trace()
    if clockwise:
        vec1 = (polygon - polygon_next)
    else:
        vec1 = (polygon_next - polygon)
    vec1 = vec1.unsqueeze(0)
    vec2 = polygon.unsqueeze(0) - points.unsqueeze(1).unsqueeze(1)
    #vec2 = vec2[..., [1, 0]]
    vec2 = vec2.flip(-1)
    vec2[..., 1] *= -1
    cross = (vec1 * vec2).sum(-1)

    return torch.all(cross > 0, dim=2)

class AttentionConstrainedLoss(nn.Module):
    def __init__(self,
                 pc_range,
                 num_class: int,
                 #: Sequence[int],
                 query_res: Sequence[int],
                 loss_weight: int = 1.0):
        super().__init__()
        self.cls_out_channels = num_class
        self.pc_range = np.asarray(pc_range)
        self.dims = self.pc_range[3:] - self.pc_range[:3]
        #self.task_id = task_id
        self.query_res = query_res
        self._loss_weight = loss_weight

        self.h, self.w = self.query_res[0], self.query_res[1]
        ww, hh = np.meshgrid(range(self.w), range(self.h))
        ww = ww.reshape(-1)
        hh = hh.reshape(-1)
        self.ww_l = torch.LongTensor(ww).to(torch.cuda.current_device())
        self.hh_l = torch.LongTensor(hh).to(torch.cuda.current_device())
        ww = torch.FloatTensor(ww).to(torch.cuda.current_device()) + 0.5
        hh = torch.FloatTensor(hh).to(torch.cuda.current_device()) + 0.5
        ww = ww / self.w * self.dims[0] + self.pc_range[0]
        hh = hh / self.h * self.dims[1] + self.pc_range[1]
        self.grids_sensor = torch.stack([ww, hh], 1).clone().detach()
        self.effective_ratio = [1.0, 6.0]

    def find_grid_in_bbx_single(self,
                                x):
        """
        find the attention grids that are enclosed by a GT bounding box
        Args:
            query_res (Sequence[int]): the size of the query feat map
            gt_bboxes (torch.Tensor, [M, ndim]): a single GT bounding boxes set for a scene
        """
        query_res, gt_bboxes = x
        #gt_bboxes = torch.from_numpy(gt_bboxes).cuda()
        bboxes_grid_ind_list = []
        if len(gt_bboxes > 0):
            temp_grid_flag = -1 * torch.ones(query_res, dtype=torch.long).cuda()
            effective_boxes = gt_bboxes[:, [0, 1, 3, 4]].clone().detach()  # [M, 4]
            effective_ratio_l = (self.dims[0] / self.w) / effective_boxes[:, 2]  # [M]
            effective_ratio_w = (self.dims[1] / self.h) / effective_boxes[:, 3]  # [M]
            effective_ratio_l = effective_ratio_l.clamp(min=self.effective_ratio[0],  # [M]
                                                        max=self.effective_ratio[1])  # [M]
            effective_ratio_w = effective_ratio_w.clamp(min=self.effective_ratio[0],  # [M]
                                                        max=self.effective_ratio[1])  # [M]
            effective_boxes[:, 2] *= effective_ratio_l
            effective_boxes[:, 3] *= effective_ratio_w
            angles = gt_bboxes[:, -1]
            effective_boxes = center_to_corner_box2d(
                effective_boxes[:, :2], effective_boxes[:, 2:4], angles)
            grid_real_centers = self.grids_sensor
            w_indices = self.ww_l
            h_indices = self.hh_l
            for i in range(len(gt_bboxes)):
                pos_mask = points_in_convex_polygon_torch(
                    grid_real_centers, effective_boxes[i].unsqueeze(0))  # [num_points, 8]
                pos_ind = pos_mask.nonzero()[:, 0]
                gt_center = gt_bboxes[i: i + 1, :2]  # [1, 2]
                dist_to_grid_center = torch.norm(grid_real_centers - gt_center, dim=1)  # [W * H]
                min_ind = torch.argmin(dist_to_grid_center)
                if min_ind not in pos_ind:
                    pos_ind = torch.cat([pos_ind.reshape(-1, 1), min_ind.reshape(-1, 1)],
                                        dim=0).reshape(-1)
                pos_h_indices = h_indices[pos_ind]  # [num_pos]
                pos_w_indices = w_indices[pos_ind]  # [num_pos]
                if len(pos_h_indices):
                    if not (temp_grid_flag[pos_h_indices, pos_w_indices] == -1).all():
                        unique_pos_h_indices = pos_h_indices.new_zeros((0,))
                        unique_pos_w_indices = pos_w_indices.new_zeros((0,))
                        for ph, pw in zip(pos_h_indices, pos_w_indices):
                            if temp_grid_flag[ph, pw] == -1:
                                unique_pos_h_indices = torch.cat(
                                    (unique_pos_h_indices, ph.view((1))))
                                unique_pos_w_indices = torch.cat(
                                    (unique_pos_w_indices, pw.view((1))))
                            else:
                                temp_grid_flag[ph, pw] = -1
                        pos_h_indices = unique_pos_h_indices
                        pos_w_indices = unique_pos_w_indices
                    temp_grid_flag[pos_h_indices, pos_w_indices] = i
            temp_grid_flag = temp_grid_flag.view(-1)
            for i in range(len(gt_bboxes)):
                bbx_grid_ind = torch.where(temp_grid_flag == i)[0]
                bboxes_grid_ind_list.append(bbx_grid_ind)
        return bboxes_grid_ind_list

    def find_grid_in_bbx(self,
                         gt_bboxes: List[np.ndarray]):
        query_sizes = [self.query_res for i in range(len(gt_bboxes))]
        map_results = map(self.find_grid_in_bbx_single, zip(query_sizes, gt_bboxes))
        return map_results

    def compute_var_loss(self,
                         atten_map: torch.Tensor,
                         grid_ind_list: List[torch.Tensor]):
        var_loss = 0.0
        var_pos_num = 0.0
        for i in range(len(grid_ind_list)):
            grid_ind = grid_ind_list[i]
            temp_var_loss = 0.0
            if len(grid_ind) > 0:
                atten_score = atten_map[grid_ind, :]
                var_t = torch.var(atten_score, 1)
                var = var_t.mean()
                temp_var_loss = temp_var_loss + (0.0 - var)
                var_pos_num += 1
            var_loss = var_loss + temp_var_loss
        return var_loss, var_pos_num

    def forward(self,
                atten_map: torch.Tensor,
                gt_bboxes: List[np.ndarray],
                gt_labels: List[np.ndarray],
                **kwargs):
        ret_dict = {}
        batch_grid_ind_list = list(self.find_grid_in_bbx(gt_bboxes))
        var_loss = torch.tensor(0.0).cuda()
        var_pos_num = 0.0
        for i in range(len(gt_bboxes)):
            grid_ind_list = batch_grid_ind_list[i]
            if len(grid_ind_list) > 0:
                temp_var_loss, temp_var_pos_num = self.compute_var_loss(
                    atten_map[i], grid_ind_list)
                var_loss = var_loss + temp_var_loss
                var_pos_num += temp_var_pos_num
        var_pos_num = max(var_pos_num, 1)
        norm_var_loss = var_loss * 1.0 / var_pos_num
        ret_dict["var_loss"] = norm_var_loss
        return ret_dict
