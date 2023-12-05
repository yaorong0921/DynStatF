import json
import numpy as np
import torch

from pyquaternion import Quaternion


from pcdet.ops.iou3d_nms.iou3d_nms_utils import nms_gpu


def perform_nms(boxes_list, scores_list, thresh=0.2):
    # merge prediction results
    ensemble_boxes = np.array(boxes_list)
    ensemble_scores = np.array(scores_list)

    box_preds = torch.from_numpy(ensemble_boxes).float().cuda()
    box_scores = torch.from_numpy(ensemble_scores).cuda()

    box_scores_nms, indices = torch.topk(box_scores, k=box_scores.shape[0])
    boxes_for_nms = box_preds[indices]
    keep_idx, selected_scores = nms_gpu(boxes_for_nms[:, 0:7], box_scores_nms, thresh=thresh)

    selected = indices[keep_idx[:83]]

    return selected.cpu().numpy()


def merge_det_res(json_list):
    nusc_annos = {
        'results': {},
        'meta': None,
    }

    nusc_annos['meta'] = {
        'use_camera': False,
        'use_lidar': True,
        'use_radar': False,
        'use_map': False,
        'use_external': False,
    }

    n_res = len(json_list)
    
    res_info_list = []

    for idx in range(n_res):
        cur_json_path = json_list[idx]

        with open(cur_json_path) as f:
            data = json.load(f)

            res_info_list.append(data['results'])

    annos_keys = res_info_list[0].keys()

    for key in annos_keys:
        boxes_list = []
        scores_list = []
        velocity_list = []
        detection_name_list = []
        attr_list = []
        quat_list = []

        annos = []

        for idx in range(n_res):
            cur_res = res_info_list[idx][key]
            n_predictions = len(cur_res)
            for cur_idx in range(n_predictions):
                q = Quaternion(np.array(cur_res[cur_idx]['rotation']))
                # Project into xy plane.
                v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

                # Measure yaw using arctan.
                yaw = np.arctan2(v[1], v[0])

                cur_box = cur_res[cur_idx]['translation'] + cur_res[cur_idx]['size'] + [yaw]
                cur_vel = cur_res[cur_idx]['velocity']
                cur_name = cur_res[cur_idx]['detection_name']
                cur_score = cur_res[cur_idx]['detection_score']
                cur_attr = cur_res[cur_idx]['attribute_name']

                boxes_list.append(cur_box)
                scores_list.append(cur_score)
                velocity_list.append(cur_vel)
                detection_name_list.append(cur_name)
                attr_list.append(cur_attr)
                quat_list.append(cur_res[cur_idx]['rotation'])

        selected = perform_nms(boxes_list, scores_list)

        for f_idx in range(len(boxes_list)):
            if f_idx in selected:
                nusc_anno = {
                    'sample_token': res_info_list[0][key][0]['sample_token'],
                    'translation': boxes_list[f_idx][:3],
                    'size': boxes_list[f_idx][3:6],
                    'rotation': quat_list[f_idx],
                    'velocity': velocity_list[f_idx],
                    'detection_name': detection_name_list[f_idx],
                    'detection_score': scores_list[f_idx],
                    'attribute_name': attr_list[f_idx]
                }

                annos.append(nusc_anno)

        nusc_annos['results'].update({key: annos})

    return nusc_annos


def merge_jsons(json_path_list, save_path):

    res_merge = merge_det_res(json_path_list)
    print(res_merge.keys())
    with open(save_path, 'w') as f:
        json.dump(res_merge, f)


if __name__ == '__main__':
    json_path_list = ['../output/cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_two_stream/test_flip/eval/epoch_20/val/0/final_result/data/results_nusc.json', \
                     '../output/cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_two_stream/test_flip/eval/epoch_20/val/1/final_result/data/results_nusc.json']
    save_path = '../output/cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_two_stream/test_flip/eval/epoch_20/val/flip_results_nusc.json'
    merge_jsons(json_path_list, save_path)
