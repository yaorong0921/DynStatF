
# DynStatF

This repo includes code of the paper ["DynStatF: An Efficient Feature Fusion Strategy for LiDAR 3D Object Detection"](https://openaccess.thecvf.com/content/CVPR2023W/E2EAD/html/Rong_DynStatF_An_Efficient_Feature_Fusion_Strategy_for_LiDAR_3D_Object_CVPRW_2023_paper.html) accepted to CVPR 2023 workshop.

#### Abstract:
In this work, we propose a novel feature fusion strategy, DynStaF (Dynamic-Static Fusion), which enhances the rich semantic information provided by the multi-frame (dynamic branch) with the accurate location information from the current single-frame (static branch). To effectively extract and aggregate complimentary features, DynStaF contains two modules, Neighborhood Cross Attention (NCA) and Dynamic-Static Interaction (DSI), operating through a dual pathway architecture. NCA takes the features in the static branch as queries and the features in the dynamic branch as keys (values). When computing the attention, we address the sparsity of point clouds and take only neighborhood positions into consideration. NCA fuses two features at different feature map scales, followed by DSI providing the comprehensive interaction. To analyze our proposed strategy DynStaF, we conduct extensive experiments on the nuScenes dataset. On the test set, DynStaF increases the performance of PointPillars in NDS by a large margin from 57.7% to 61.6%. When combined with CenterPoint, our framework achieves 61.0% mAP and 67.7% NDS.


## Requirements
Please refer to the `requirements.txt` and `myEnvBkp.txt` for for detailed information about the environment setup.

All experiments were conducted using eight V100 GPUs.

## Experiment Code
#### Prerequisites
- Dataset: Please follow the instructions on [nuScenes](https://www.nuscenes.org/nuscenes) to download the dataset (Full dataset (v1.0)). After downloading, change the data path in `tools/cfgs/dataset_configs/nuscenes_data.yaml`.
- OpenPCDet Toolkit: Please follow the instructions on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) to install the OpenPCDet toolbox. 
- NATTEN module: Please build this module following the instructions at [NATTEN](https://github.com/SHI-Labs/NATTEN). 

### Run Scripts
To train DynStaF with the PointPillar backbone, please run ```bash ./scripts/dist_train.sh 8 --cfg_file ./cfgs/nuscenes_models/cbgs_pp_multihead_dynstaf.yaml```

To train with the CenterPoint backbone (voxel 0.075m), please run
```bash ./scripts/dist_train.sh 8 --cfg_file ./cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_dynstaf.yaml```

To test the trained model, please run e.g.,
```python test.py --cfg_file ./cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_dynstaf.yaml --batch_size 4 --workers 8  --ckpt [PATH TO /ckpt/checkpoint_epoch_20.pth]```

### Code Structure
- In addition to the two main configs for DynStaF (`cbgs_voxel0075_res3d_centerpoint_dynstaf.yaml` and `cbgs_pp_multihead_dynstaf.yaml`), there are addtional configs available for various ablation models. Please refer to the directory`tools/cfgs/nuscenes_models/`. 
- Implementation detials of NCA can be found in `models/backbones_2d/base_bev_two_stream_nat_backbone.py` (`BaseBEVTwoStreamNATBackbone`). 
- Implementation detaisl of DSI can be found in `models/dense_heads/center_head.py` for CenterPoint and `models/dense_heads/anchor_head_multi.py` for PointPillar. Other fusion strategy such as CBAM or NCA only used for ablation studies are in comments. 



## Citation 
If you use the CUB-GHA dataset or code in this repo in your research, please cite:

```
@inproceedings{rong2023dynstatf,
  title={DynStatF: An Efficient Feature Fusion Strategy for LiDAR 3D Object Detection},
  author={Rong, Yao and Wei, Xiangyu and Lin, Tianwei and Wang, Yueyu and Kasneci, Enkelejda},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3237--3246},
  year={2023}
}
```

## Acknowledgement
We thank the following works providing helpful components in our work:
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/tree/master)
- [NAT](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer)
- [CBAM](https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py)
- Shen et al., "Cross Attention-guided Dense Network for Images Fusion", 2021.

Contact me (yao.rong@tum.de) if you have any questions or suggestions.


