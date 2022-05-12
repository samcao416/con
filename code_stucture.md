# Datasets
    config.get_datset:
        method: conv_onet
        dataset_type: Shapes3D (no else is coded)
        dataset_folder: the root path of data
        categories: class names (subfoler name)
        splits: {
                    'train' : 'train',
                    'val' : 'val',
                    'test' : 'test'
                }
        input_type: pointcloud / partial_pointcloud / img / pointcloud_crop
        
        field: {
            'points' :
                instance of PointsField or
                            PatchPointsFields (pointcloud_crop)
                    PointsField (points_file(points file directory name),
                                 points_transform: the instance of SubsamplePoints,
                                 points_unpackbits,
                                 multi_files):
                        data :  {
                                    None : points,
                                    'occ': occupancies
                                }
                    PatchPointsFields add some operations other than PointFields
                        data :  {   
                                    None : points,
                                    'occ': occupancies
                                    'normalized': p_n
                                } 
                            TODO : figure out the operations and p_n

            'points_iou' (for val & test):
                similar to 'points', but the points file directory name is points_iou_file

            'voxels' (if voxel_file exists): 
                pass


            'inputs_fields'(if inputs_fields is not None): 
                inputs_fields:
                    input_type: None:
                        None
                    input_type: pointcloud:
                        an instance of PointCloudField
                    input_type: partial_pointcloud:
                        an instance of PartialPointCloudField
                    input_type: pointcloud_crop:
                        an instance of PatchPointCloudField
                    input_type: voxels:
                        an instance of VoxelsField

        
            'idx' (if return_idx is true(val_dataset)):
                instance of class IndexField
        }

        Difference between PointsField,
                           PatchPointsField,
                           PointCloudField,
                           PartialPointCloudField,
                           PatchPointCloudField:
            fields: PointsField, PatchPointsField
            input_field: PointCloudField, PartialPointCloudField, PatchPointCloudField

            unpackbits only in fields

            data (fields) :
            {
                None: points (if dtype is float16, turn it into float32 and add a Gaussian noise with σ = 1e-4),
                'occ': occupancies
            }
            data (input_fields):{
                None: points,
                'normals': normals
            }
        
        
        dataset: an instance of class Shape3dDataset:
            self.fields: field dict(field_name: field) i.e. 'points': PointsField...
            categories : subfolder names
            self.metadata (dict):
                {
                    c (class name):
                    {
                        'id': c (subfolder names / class name),
                        'name': content names(if metadata.yaml exist) / 'n/a'
                        'idx': index of c
                    }
                }
            self.models (list):
                [
                    {
                        'category': c (subfolder names / class name),
                        'model': models_c[i] (sub-subfolder id)
                    }
                ]
            self.depth

            return data: a dict:
                {
                    'points' : points,
                    'points.occ' : occupancies,
                    ('points_iou' : points,)
                    ('points_iou.occ' : occupancies,) (exists only for val & test)
                    'input_fields' : points,
                    'input_fields.normals': normals
                    'index_fields' : index
                }


    Data folder structure (Synthetic Room):
    dataset_folder (data/path) /
        categories (class) /
            group ID /
                pointcloud /
                    pointcloud_00.npz
                    ...
                points_iou /
                    points_ios_00.npz
                    ...
                item_dict.npz
                pointcloud0.ply
            test.lst
            train.lst
            val.lst

    Data folder structure (Shapenet):
    dataset_folder (data/path) /

# Model
    config.get_model
    method = 'conv_onet'
    model: an instance of class method_dict['conv_onet'].config.get_model:
        encoder: resnet18 / pointnet_local_pool
        decoder: simple_local / simple_local_crop / simple_local_point
        dim: 3
        c_dim: 32
        encoder_kwargs:
            {
                'hidden_dim': 32
                'plane_type': 'grid' / ['xz']
                'grid_resolution' / 'plane_resolution': 32 / 128
                'unet3d' / 'unet': True
                'unet3d_kwargs' / 'unet_kwargs': 
                    {
                        'num_levels': 3
                        'f_maps': 32
                        'in_channels': 32
                        'out_channels': 32
                    }
                /
                    {
                        'depth': 5
                        'merge_mode': concat
                        'start_filts': 32
                    }
                
                for pointcloud_crop:
                'unit_size': 0.005
            }
        decoder_kwargs:
            {
                'sample_mode': bilinear / nearest
                'hidden_size': 32
                for pointcloud_crop:
                'unit_size': 0.005
            }
        padding: 0.1

        decoder:
            An instance of class LocalDecoder / PatchLocalDecoder / LocalPointDecoder

            self.fc_c:
                [
                    nn.Linear(c_dim(32), hidden_size(256)) * n_blocks(5)
                ]
            self.fc_p:
                nn.Linear(dim(3), hidden_size(256))
            self.blocks:
                [
                    ResnetBlockFC(hidden_size(256)) * n_blocks(5)
                ]
            self.fc_out:
                nn.Linear(hidden_size(256), 1)

            self.actvn = F.relu
            self.sample_mode = nearest / bilinear
            self.padding = 0.005


            forward:
            c: sampled feature
            p: points

            net:
            {
                p -> nn.Linear(3, 256) -> net

                for i in range(5):
                    c -> nn.Linear(32, 256) -> c_out
                    c_out + net -> ResnetBlockFC(256) -> net
                
                net -> F.relu -> net_activated
                net_activated -> nn.Linear(256, 1) -> out

            }
            out.squeeze(-1)

        encoder:
            An instance of pointnet.LocalPoolPointnet /
                           pointnet.PatchLocalPoolPointnet /
                           pointnetpp.PointNetPlusPlus /
                           voxels.LocalVoxelEncoder
            
            self.fc_pos =  nn.linear(dim(3), 2 * hidden_dim (2 * 128 = 256))
            self.blocks:
                [
                    ResnetBlockFC( 2* hidden_dim(256), hidden_dim(128)) * n_blocks(5)
                ]
            self.fc_c:
                nn.Linear(hidden_dim(128), c_dim(32))
            self.actvn = nn.ReLU()

            if unet:
                self.unet:
                    Unet(c_dim(32), in_channels = c_dim(32), **unet_kwargs:
                                                                    depth: 5
                                                                    merge_mode: concat
                                                                    start_filts: 32)
            
            if unet3d:
                self.unet3d = UNet3D(**unet3d_kwargs:
                                            num_levels: 3
                                            f_maps: 32
                                            in_channels: 32
                                            out_channels: 32)
            self.reso_plane: 128 / None
            self.reso_grid : None / 32
            plane_typeL ['xz'] / grid
            self.padding = 0.005

            scatter_type = 'max' : scatter_max

            forward:
                coord: normalized_coordinate
                index: coordinate2index
                p: points

                net:
                {
                    p -> nn.Linear(3, 256) -> net
                    net -> ResnetBlockFC(256, 128) -> net
                    for i in range(5 - 1):
                        pooled = pool_local(coor, index, net)
                        net = torch.cat([net. pooled], dim = 2)
                        net -> ResnetBlockFC(256, 128) -> net
                }

                net -> nn.Linear(128, 32) -> c

                generate_plane_features: unet
                generate_grid_features: unet3d

        ResnetBlockFC:
            size_in (256)
            size_out (256 / 128)
            size_h = min(size_in, size_out)

            self.fc_0 = nn.Linear(size_in, size_h)
            self.fc_1 = nn.Linear(size_h, size_out)
            self.actvn = nn.ReLU()

            if size_in != size_out:
                self.shortcut:
                    nn.Linear(size_in (256), size_out (256))
                
            net:
                x -> nn.ReLU -> nn.Linear(256, 256 / 128) -> net
                net -> nn.ReLU -> nn.Linear(256 / 128, 256/ 128) -> dx
                if size_in != size_out:
                    x -> nn.Linear(256, 128) -> x_s
                else:
                    x -> x_s
            final_out: x_s + dx
        
        UNet3D:
            in_channels: 32
            out_channels: 32
            
                
# Generator (code to generate mesh)
    config.get_generator
    method: conv_net
    generator: an instance of method_dict['conv_net'].config.get_generator()

# Trainer
    config.get_trainer
    method_dict['conv_net'].config.get_trainer
    threshold: 0.2
    eval_sample: False

    trainer: an instance of training.Trainer:
        compute_loss:
            F.binary_cross_entropy_with_logits


