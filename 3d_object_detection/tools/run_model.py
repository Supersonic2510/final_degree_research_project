import argparse
import glob
from pathlib import Path
import numpy as np
import torch
import json

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
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


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--output_dir', type=str, default='../output_predictions', help='specify the output directory for results')
    parser.add_argument('--onnx_export_path', type=str, default='model.onnx', help='specify the path for exporting the model to ONNX')

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

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            data_dict = demo_dataset.collate_batch([data_dict])
            data_dict['frame_id'] = torch.tensor(data_dict['frame_id'], dtype=torch.float32).cuda()
            load_data_to_gpu(data_dict)

            pred_dicts, _ = model.forward(data_dict)

            # Ensure output directory exists
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            json_dict = {
                "ref_boxes": pred_dicts[0]['pred_boxes'].cpu().tolist(),
                "ref_scores": pred_dicts[0]['pred_scores'].cpu().tolist(),
                "ref_labels": pred_dicts[0]['pred_labels'].cpu().tolist()
            }

            # Serialize the dictionary to a JSON formatted string
            json_str = json.dumps(json_dict, indent=4)

            # Save predictions to JSON file
            output_file = output_dir / f'predictions_{idx}.json'
            with open(output_file, 'w') as json_file:
                json_file.write(json_str)

            logger.info(f'Saved predictions for sample {idx} to {output_file}')


if __name__ == '__main__':
    main()
