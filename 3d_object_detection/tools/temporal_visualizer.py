import argparse
from pathlib import Path
import open3d as o3d
import cv2
import numpy as np

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network
from pcdet.utils import common_utils

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        data_file_list = list(Path(root_path).rglob('*.bin'))
        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        input_dict = {
            'points': points,
            'frame_id': index,
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--samples', type=int, default=10, help='specify the number of samples to process')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg

def load_model(cfg, checkpoint_path, logger, dataset=None):
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=checkpoint_path, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    return model

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    num_samples = min(args.samples, len(demo_dataset))

    model = load_model(cfg, args.ckpt, logger, demo_dataset)

    for idx in range(num_samples):
        logger.info(f'Visualized sample index: \t{idx + 1}')
        data_dict = demo_dataset.collate_batch([demo_dataset[idx]])
        pred_dicts = predict(model, data_dict)

        lidar_points = data_dict['points'][:, 1:].cpu().numpy()

        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().point_size = 1.0
        vis.get_render_option().background_color = np.zeros(3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar_points)
        vis.add_geometry(pcd)

        V.draw_scenes(
            points=data_dict['points'][:, 1:], 
            ref_boxes=pred_dicts[0]['pred_boxes'],
            ref_scores=pred_dicts[0]['pred_scores'], 
            ref_labels=pred_dicts[0]['pred_labels']
        )

        vis.run()
        vis.destroy_window()

if __name__ == '__main__':
    main()
