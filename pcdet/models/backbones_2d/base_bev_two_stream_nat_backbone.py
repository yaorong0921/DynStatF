import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from ..fusion_module.nat import NATBlock


class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), #kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvTokenizer_Layer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=(1, 1), stride=(1, 1))
            # nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x



class BaseBEVTwoStreamNATBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        # # ### layers to arrange dim for summation single and mutliframe features
        # self.reduce_layers = nn.ModuleList()

        ### nat layers after each block for fusion
        self.fusion_nats = nn.ModuleList()

        ### tokenizer layers for fusion nats
        self.single_tokenizers = nn.ModuleList()
        self.multi_tokenizers = nn.ModuleList()

        ### layers to upsample the features maps
        self.upsample_layers = nn.Sequential(
                        nn.ConvTranspose2d(input_channels*2, input_channels,
                            kernel_size=(2, 2), stride=(2, 2)),
                        nn.ConvTranspose2d(input_channels, input_channels,
                            kernel_size=(2, 2), stride=(2, 2)),
                        nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
                        nn.ReLU())

        for idx in range(num_levels):
            # # FOR 3D Long: add the first block to reduce the dimansion to get it the same as the 3d backbone
            # if idx==0: 
            #     cur_layers = [
            #         nn.ZeroPad2d(1),
            #         nn.Conv2d(
            #             c_in_list[idx], num_filters[idx], kernel_size=3,
            #             stride=layer_strides[idx], padding=0, bias=False
            #         ),
            #         nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
            #         nn.ReLU()
            #     ]

            #     for k in range(layer_nums[idx]):
            #         cur_layers.extend([
            #             nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
            #             nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
            #             nn.ReLU()
            #         ])
            #     self.blocks.append(nn.Sequential(*cur_layers))
            # ## here are the original FOR 2D
            # else: 
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]

            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

                # ### layers to arrange dim for summation single and mutliframe features
                # channel_reduce = [
                #             nn.Conv2d(num_filters[idx]*2, num_filters[idx], 1, stride=1, bias=False),
                #             nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                #             nn.ReLU()]
                # self.reduce_layers.append(nn.Sequential(*channel_reduce))

                ### nat layers after each block for fusion
                self.single_tokenizers.append(ConvTokenizer_Layer(in_chans=num_filters[idx], embed_dim=num_filters[idx]//2, norm_layer=nn.LayerNorm))
                self.multi_tokenizers.append(ConvTokenizer_Layer(in_chans=num_filters[idx], embed_dim=num_filters[idx]//2, norm_layer=nn.LayerNorm))
                self.fusion_nats.append(NATBlock(dim=num_filters[idx]//2, depth=1, depth_cross=1, num_heads=8, kernel_size=7, downsample=False))

                # #### for rebuttal, no self atten.
                # self.single_tokenizers.append(ConvTokenizer_Layer(in_chans=num_filters[idx], embed_dim=num_filters[idx], norm_layer=nn.LayerNorm))
                # self.multi_tokenizers.append(ConvTokenizer_Layer(in_chans=num_filters[idx], embed_dim=num_filters[idx], norm_layer=nn.LayerNorm))
                # self.fusion_nats.append(NATBlock(dim=num_filters[idx], depth=1, depth_cross=1, num_heads=8, kernel_size=7, downsample=False))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

        # NAT block for fusion at the beginning
        self.patch_embed_single = ConvTokenizer(
            in_chans=input_channels, embed_dim=input_channels, norm_layer=nn.LayerNorm)

        self.patch_embed_multi = ConvTokenizer(
            in_chans=input_channels, embed_dim=input_channels, norm_layer=nn.LayerNorm)

        # self.patch_embed_single = ConvTokenizer_Layer(
        #     in_chans=input_channels, embed_dim=input_channels, norm_layer=nn.LayerNorm)
        # self.patch_embed_multi = ConvTokenizer_Layer(
        #     in_chans=input_channels, embed_dim=input_channels, norm_layer=nn.LayerNorm)


        self.nat_block = NATBlock(dim=input_channels, depth=3, depth_cross=1, num_heads=8, kernel_size=7, downsample=False)

        # ## to reduce W, H for merging into 3d outputs
        # self.f_reduce = nn.Sequential(
        #                     nn.Conv2d(
        #                         input_channels, input_channels,
        #                         3,
        #                         stride=3, bias=False, padding=1
        #                     ),
        #                     nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
        #                     nn.ReLU()
        #             )

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
# ###### for ablation
#         spatial_features = data_dict['spatial_features']
#         ups = []
#         ret_dict = {}

#         # x = spatial_features
#         # ### reduce the height and width
#         # spatial_features = self.reduce_layers(spatial_features)
#         x = spatial_features

#         for i in range(len(self.blocks)):
#             x = self.blocks[i](x)

#             stride = int(spatial_features.shape[2] / x.shape[2])
#             # # rec_dict['spatial_features_%dx' % stride] = x
#             data_dict['spatial_features_%dx' % stride] = x

#             # fuse features from single-frame branch (lateral connection)
#             # TODO: try different strategies for fusion (concat + conv)
#             # Strategy1: add features directly
#             # x += self.reduce_layers[i](data_dict['cur_spatial_features_%dx' % stride])

#             if len(self.deblocks) > 0:
#                 ups.append(self.deblocks[i](x))
#             else:
#                 ups.append(x)

#         if len(ups) > 1:
#             x = torch.cat(ups, dim=1)
#         elif len(ups) == 1:
#             x = ups[0]

#         if len(self.deblocks) > len(self.blocks):
#             x = self.deblocks[-1](x)

#         data_dict['spatial_features_2d'] = x
# ###################

        spatial_features = data_dict['cur_spatial_features']
        spatial_features_multi = data_dict['spatial_features']
        # print('Single feature map: ', spatial_features.shape)
        # print('Multiple feature map: ', spatial_features_multi.shape)

        # ## FOR 3D: reduce the feature of single branch
        # spatial_features = self.f_reduce(spatial_features)
        # # spatial_features = self.blocks[0](spatial_features)
        # # print(spatial_features.shape, spatial_features_multi.shape)

        spatial_features_embed_single = self.patch_embed_single(spatial_features)
        spatial_features_embed_multi = self.patch_embed_multi(spatial_features_multi)

        # ### coords: [batch_index, z=0, y, x] (https://blog.csdn.net/qq_41366026/article/details/123006401)
        # coords = deepcopy(data_dict['cur_voxel_coords']).long()
        # #torch.set_printoptions(profile="full")
        # coords_t = torch.cat([coords[:,2].unsqueeze(dim=1), coords[:,3].unsqueeze(dim=1)], dim=1)
        # x_mask = self.coord_to_mask(coords_t, torch.zeros_like(spatial_features_embed_single).to(spatial_features_embed_single.get_device()), rescale=4)

        # spatial_features = self.nat_block(spatial_features_embed_single, spatial_features_embed_multi, x_mask=x_mask)
        spatial_features = self.nat_block(spatial_features_embed_single, spatial_features_embed_multi)

        ups = []
        ret_dict = {}
        spatial_features = self.upsample_layers(spatial_features.permute(0, 3, 1, 2))
        x = spatial_features

        # # FOR 3D Long: 
        # for i in range(len(self.blocks)-1):
        #     x = self.blocks[i+1](x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stride = int(spatial_features.shape[2] / x.shape[2])

            # fuse features from multi-frame branch (lateral connection)
            # TODO: try different strategies for fusion (concat + conv)
            # Strategy1: add features directly
            # x = x.clone() + data_dict['spatial_features_%dx' % stride] #self.reduce_layers[i](data_dict['spatial_features_%dx' % stride])

            # # Strategy2: add nat feature fusion
            x = self.fusion_nats[i](self.single_tokenizers[i](x), self.multi_tokenizers[i](data_dict['spatial_features_%dx' % stride])) #, x_mask=None
            x = x.permute(0, 3, 1, 2)

            ret_dict['spatial_features_%dx_single' % stride] = x

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d_single'] = x

        return data_dict
    

    def coord_to_mask(self, coord, zero_tensor, rescale=4):
        ### coord: [y, x] ## zero_tensor size torch.Size([4, 128, 128, 64])
        zero_tensor[:, coord[:, 0]//rescale, coord[:, 1]//rescale, :] = 1
        return zero_tensor

