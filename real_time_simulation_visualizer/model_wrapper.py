import torch
from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
import numpy as np
from collections import defaultdict
import cumm.tensorview as tv


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    @staticmethod
    def prepare_data(points):
        def set_lidar_aug_matrix(data_dict):
            lidar_aug_matrix = np.eye(4)
            data_dict['lidar_aug_matrix'] = lidar_aug_matrix
            return data_dict

        def point_encoder(data_dict):
            data_dict['use_lead_xyz'] = True
            return data_dict

        def data_processor(data_dict):
            def mask_points_and_boxes_outside_range(data_dict):
                point_cloud_range_voxelnext = [-75.2, -75.2, -5.0, 75.2, 75.2, 3.0]

                mask = ((data_dict['points'][:, 0] >= point_cloud_range_voxelnext[0])
                        & (data_dict['points'][:, 0] <= point_cloud_range_voxelnext[3])
                        & (data_dict['points'][:, 1] >= point_cloud_range_voxelnext[1])
                        & (data_dict['points'][:, 1] <= point_cloud_range_voxelnext[4]))
                data_dict['points'] = data_dict['points'][mask]
                return data_dict

            def transform_points_to_voxels(data_dict):
                voxel_size_voxelnext = [0.1, 0.1, 0.2]
                point_cloud_range_voxelnext = np.array([-75.2, -75.2, -5.0, 75.2, 75.2, 3.0])
                num_point_features_voxelnext = 4
                max_num_points_per_voxel_voxelnext = 5
                max_num_voxels_voxelnext = 60000

                voxel_generator = VoxelGenerator(
                    vsize_xyz=voxel_size_voxelnext,
                    coors_range_xyz=point_cloud_range_voxelnext,
                    num_point_features=num_point_features_voxelnext,
                    max_num_points_per_voxel=max_num_points_per_voxel_voxelnext,
                    max_num_voxels=max_num_voxels_voxelnext
                )

                points = data_dict['points']
                voxel_output = voxel_generator.point_to_voxel(tv.from_numpy(points))
                tv_voxels, tv_coordinates, tv_num_points = voxel_output

                voxels = tv_voxels.numpy()
                coordinates = tv_coordinates.numpy()
                num_points = tv_num_points.numpy()

                data_dict['voxels'] = voxels
                data_dict['voxel_coords'] = coordinates
                data_dict['voxel_num_points'] = num_points

                return data_dict

            data_dict = mask_points_and_boxes_outside_range(data_dict)
            data_dict = transform_points_to_voxels(data_dict)

            return data_dict

        data_dict = {
            'points': points,
            'frame_id': 0,
        }

        data_dict = set_lidar_aug_matrix(data_dict)
        data_dict = point_encoder(data_dict)
        data_dict = data_processor(data_dict)

        return data_dict

    @staticmethod
    def load_data_to_gpu_custom(batch_dict):
        for key, val in batch_dict.items():
            if not isinstance(val, np.ndarray):
                continue
            elif key in ['frame_id']:
                continue
            else:
                batch_dict[key] = torch.from_numpy(val).float().cuda()
        return batch_dict

    @staticmethod
    def collate_batch(batch_list):
        data_dict = defaultdict(list)

        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        batch_size_ratio = 1

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    if isinstance(val[0], list):
                        batch_size_ratio = len(val[0])
                        val = [i for item in val for i in item]
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    if isinstance(val[0], list):
                        val = [i for item in val for i in item]
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size * batch_size_ratio
        return ret

    def forward(self, points):
        data_dict = self.prepare_data(points)
        data_dict = self.collate_batch([data_dict])
        data_dict['frame_id'] = torch.tensor(data_dict['frame_id'], dtype=torch.float32).cuda()

        data_dict = self.load_data_to_gpu_custom(data_dict)

        output = self.model.forward(data_dict)
        first_item = output[0]

        pred_boxes = first_item[0]['pred_boxes']
        pred_scores = first_item[0]['pred_scores']
        pred_labels = first_item[0]['pred_labels']
        pred_ious = first_item[0]['pred_ious'][0]

        result_dict = {
            'pred_boxes': pred_boxes,
            'pred_scores': pred_scores,
            'pred_labels': pred_labels,
            'pred_ious': pred_ious
        }

        return result_dict