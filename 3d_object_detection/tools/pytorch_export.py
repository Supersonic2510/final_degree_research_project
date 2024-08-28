import argparse
import glob
from pathlib import Path
import numpy as np
import torch
import json

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder
from pcdet.models import build_network, load_data_to_gpu
from pcdet.models.detectors import VoxelNeXt
from pcdet.utils import common_utils

from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
import cumm.tensorview as tv
from collections import defaultdict
from easydict import EasyDict as edict


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


class ModelDataPreparation:
    def __init__(self, root_path=None):
        self.root_path = root_path
        self.points = np.fromfile(Path(root_path), dtype=np.float32).reshape(-1, 4)
        self.dataset_cfg = edict({
            'POINT_CLOUD_RANGE': [-75.2, -75.2, -5.0, 75.2, 75.2, 3.0],
            'POINT_FEATURE_ENCODING': {
                'encoding_type': 'absolute_coordinates_encoding',
                'used_feature_list': ['x', 'y', 'z', 'intensity'],
                'src_feature_list': ['x', 'y', 'z', 'intensity']
            },
            'DATA_PROCESSOR': [
                {
                'NAME': 'mask_points_and_boxes_outside_range',
                'REMOVE_OUTSIDE_BOXES': True
            },
            {
                                    'NAME': 'shuffle_points',
                                    'SHUFFLE_ENABLED': {
                                        'train': True,
                                        'test': False
                                    }
            },
             {
                                    'NAME': 'transform_points_to_voxels',
                                    'VOXEL_SIZE': [0.1, 0.1, 0.2],
                                    'MAX_POINTS_PER_VOXEL': 5,
                                    'MAX_NUMBER_OF_VOXELS': {
                                        'train': 60000,
                                        'test': 60000
             }
            }
             ],
            '_BASE_CONFIG_': 'cfgs/dataset_configs/once_dataset.yaml'
        })

        self.training = False
        self.class_names = ['Car', 'Bus', 'Truck', 'Pedestrian', 'Cyclist']
        self.logger = None

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_augmentor = None
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False
        self.depth_downsample_factor = None

        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )

        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )

    def prepare_data(self):
        def set_lidar_aug_matrix(data_dict):
            lidar_aug_matrix = np.eye(4)
            data_dict['lidar_aug_matrix'] = lidar_aug_matrix
            return data_dict

        def point_encoder(data_dict):
            data_dict['use_lead_xyz'] = True
            return data_dict

        def data_processor(data_dict):
            def mask_points_and_boxes_outside_range(data_dict):

                mask = ((data_dict['points'][:, 0] >= self.point_cloud_range[0])
                        & (data_dict['points'][:, 0] <= self.point_cloud_range[3])
                        & (data_dict['points'][:, 1] >= self.point_cloud_range[1])
                        & (data_dict['points'][:, 1] <= self.point_cloud_range[4]))
                data_dict['points'] = data_dict['points'][mask]
                return data_dict

            def transform_points_to_voxels(data_dict):

                num_point_features_voxelnext = 4
                max_num_points_per_voxel_voxelnext = 5
                max_num_voxels_voxelnext = 60000

                voxel_generator = VoxelGenerator(
                    vsize_xyz=self.voxel_size,
                    coors_range_xyz=self.point_cloud_range,
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
            'points': self.points,
            'frame_id': 0,
        }

        data_dict = set_lidar_aug_matrix(data_dict)
        data_dict = point_encoder(data_dict)
        data_dict = data_processor(data_dict)

        return data_dict

    def collate_batch(self, batch_list):
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


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--output_dir', type=str, default='../output_predictions',
                        help='specify the output directory for results')
    parser.add_argument('--onnx_export_path', type=str, default='../export_model/model.onnx',
                        help='specify the path for exporting the model to ONNX')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


# Helper function to convert to JSON-serializable format
def convert_to_json_serializable(data):
    if isinstance(data, dict):
        return {key: convert_to_json_serializable(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [convert_to_json_serializable(item) for item in data]
    elif isinstance(data, (int, float, str, bool)):
        return data
    elif hasattr(data, 'shape'):
        return {"type": type(data).__name__, "shape": list(data.shape)}
    else:
        return str(data)


# Define a wrapper function to handle dictionary input for ONNX export
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
        return pred_boxes, pred_scores, pred_labels, pred_ious


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Run Model VoxelNext OpenPCDet-------------------------')

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )

    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    # Wrap the original model
    model_wrapper = ModelWrapper(model)

    model_wrapper.cuda()
    model_wrapper.eval()

    torch.save(model_wrapper, '../export_model/voxelnext_model_last.pth')


if __name__ == "__main__":
    main()
