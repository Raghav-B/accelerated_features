import types

import argparse
import torch
import torch.nn.functional as F
import onnx
import onnxsim
import os
from onnxconverter_common import float16
import openvino as ov

from modules.xfeat import XFeat
from modules.lighterglue import LighterGlue


class CustomInstanceNorm(torch.nn.Module):
    def __init__(self, epsilon=1e-5):
        super(CustomInstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), unbiased=False, keepdim=True)
        return (x - mean) / (std + self.epsilon)


def preprocess_tensor(self, x):
    return x, 1.0, 1.0 # Assuming the width and height are multiples of 32, bypass preprocessing.

def match_xfeat_star(self, mkpts0, feats0, sc0, mkpts1, feats1, sc1):
    out1 = {
        "keypoints": mkpts0,
        "descriptors": feats0,
        "scales": sc0,
    }
    out2 = {
        "keypoints": mkpts1,
        "descriptors": feats1,
        "scales": sc1,
    }

    #Match batches of pairs
    idx0_b, idx1_b = self.batch_match(out1['descriptors'], out2['descriptors'] )

    #Refine coarse matches
    match_mkpts, batch_index = self.refine_matches(out1, out2, idx0_b, idx1_b, fine_conf = 0.25)

    return match_mkpts, batch_index


def parse_args():
    parser = argparse.ArgumentParser(description="Export XFeat/Matching model to ONNX.")
    parser.add_argument(
        "--xfeat_only_model",
        action="store_true",
        help="Export only the XFeat model.",
    )
    parser.add_argument(
        "--xfeat_only_model_dualscale",
        action="store_true",
        help="Export only the XFeat dualscale model.",
    )
    parser.add_argument(
        "--xfeat_only_matching",
        action="store_true",
        help="Export only the matching.",
    )
    parser.add_argument(
        "--split_instance_norm",
        action="store_true",
        help="Whether to split InstanceNorm2d into '(x - mean) / (std + epsilon)', due to some inference libraries not supporting InstanceNorm, such as OpenVINO.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=640,
        help="Input image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Input image width.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Keep best k features.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic axes.",
    )
    parser.add_argument(
        "--export_path",
        type=str,
        default="onnx_weights/extractor.onnx",
        help="Path to export ONNX model.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset version.",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # if args.dynamic:
    #     args.height = 640
    #     args.width = 640
    # else:
    #     assert args.height % 32 == 0 and args.width % 32 == 0, "Height and width must be multiples of 32."

    if args.top_k > 4800:
        print("Warning: The current maximum supported value for TopK in TensorRT is 3840, which coincidentally equals 4800 * 0.8. Please ignore this warning if TensorRT will not be used in the future.")

    batch_size = 1
    x1 = torch.randn(batch_size, 3, 320, 320, device='cpu')
    x2 = torch.randn(batch_size, 1, args.height, args.width, device='cpu')

    xfeat = XFeat()
    xfeat.top_k = args.top_k

    if args.split_instance_norm:
        xfeat.net.norm = CustomInstanceNorm()

    xfeat = xfeat.cpu().eval()
    xfeat.dev = "cpu"

    if not args.dynamic:
        # Bypass preprocess_tensor
        xfeat.preprocess_tensor = types.MethodType(preprocess_tensor, xfeat)

    if args.xfeat_only_model:
        dynamic_axes = {"images": {0: "batch", 2: "height", 3: "width"}}
        torch.onnx.export(
            xfeat.net,
            (x1),
            args.export_path,
            verbose=False,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["feats", "keypoints", "heatmaps"],
            dynamic_axes=dynamic_axes if args.dynamic else None,
        )
    elif args.xfeat_only_model_dualscale:
        xfeat.forward = xfeat.detectAndComputeDense
        dynamic_axes = {"images": {0: "batch", 2: "height", 3: "width"}}
        torch.onnx.export(
            xfeat,
            (x1, args.top_k),
            args.export_path,
            verbose=False,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["mkpts", "feats", "sc"],
            dynamic_axes=dynamic_axes if args.dynamic else None,
        )
    elif args.xfeat_only_matching:
        xfeat.forward = types.MethodType(match_xfeat_star, xfeat)

        mkpts0 = torch.randn(batch_size, args.top_k, 2, dtype=torch.float32, device='cpu')
        mkpts1 = torch.randn(batch_size, args.top_k, 2, dtype=torch.float32, device='cpu')
        feats0 = torch.randn(batch_size, args.top_k, 64, dtype=torch.float32, device='cpu')
        feats1 = torch.randn(batch_size, args.top_k, 64, dtype=torch.float32, device='cpu')
        sc0 = torch.randn(batch_size, args.top_k, dtype=torch.float32, device='cpu')
        sc1 = torch.randn(batch_size, args.top_k, dtype=torch.float32, device='cpu')

        dynamic_axes = {
            "mkpts0": {0: "batch", 1: "num_keypoints"},
            "feats0": {0: "batch", 1: "num_keypoints", 2: "descriptor_size"},
            "sc0": {0: "batch", 1: "num_keypoints"},
            "mkpts1": {0: "batch", 1: "num_keypoints"},
            "feats1": {0: "batch", 1: "num_keypoints", 2: "descriptor_size"},
            "sc1": {0: "batch", 1: "num_keypoints"},
        }
        torch.onnx.export(
            xfeat,
            (mkpts0, feats0, sc0, mkpts1, feats1, sc1),
            args.export_path,
            verbose=False,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["mkpts0", "feats0", "sc0", "mkpts1", "feats1", "sc1"],
            output_names=["matches", "batch_indexes"],
            dynamic_axes=dynamic_axes if args.dynamic else None,
        )
    else:
        # xfeat.forward = xfeat.match_xfeat_star
        # dynamic_axes = {"images0": {0: "batch", 2: "height", 3: "width"}, "images1": {0: "batch", 2: "height", 3: "width"}}
        # torch.onnx.export(
        #     xfeat.net,
        #     x1,
        #     args.export_path,
        #     verbose=False,
        #     opset_version=14,
        #     do_constant_folding=False,
        #     input_names=["im"],
        #     output_names=["feats", "keypoints", "heatmaps"]
        # )

        lighterglue = LighterGlue()
        lighterglue = lighterglue.cpu().eval()
        lighterglue.dev = "cpu"

        # import pickle

        # with open("tensor.pkl", "rb") as f:
            # dat = pickle.load(f)

        img_size = torch.tensor((320, 320), dtype=torch.int32, device='cpu')
        img_size = img_size.unsqueeze(-2)

        data = {
            'keypoints0': torch.zeros(1, args.top_k, 2, dtype=torch.float32, device='cpu'),
            'keypoints1': torch.zeros(1, args.top_k, 2, dtype=torch.float32, device='cpu'),
            'descriptors0': torch.randn(1, args.top_k, 64, dtype=torch.float32, device='cpu'),
            'descriptors1': torch.randn(1, args.top_k, 64, dtype=torch.float32, device='cpu'),
            'image_size0': img_size,
            'image_size1': img_size,
		}

        # data = {
        #     'keypoints0': dat["keypoints"].unsqueeze(-3)[:, :, :2],
        #     'keypoints1': dat["keypoints"].unsqueeze(-3)[:, :, :2],
        #     'descriptors0': dat['descriptors'].unsqueeze(-3),
        #     'descriptors1': dat['descriptors'].unsqueeze(-3),
        #     'image_size0': img_size,
        #     'image_size1': img_size,
		# }

        torch.onnx.export(
            lighterglue,
            (data, 0.1),
            args.export_path,
            dynamo=True,
            input_names=[
                "keypoints0", "keypoints1", "descriptors0", "descriptors1", "image_size0", "image_size1"
            ],
            output_names=[
                "log_assignment", "matches0", "matches1", "matching_scores0", "matching_scores1"
            ],
        )

    model_onnx = onnx.load(args.export_path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # model_onnx, check = onnxsim.simplify(model_onnx)
    # model_onnx = float16.convert_float_to_float16(model_onnx)
    # assert check, "assert check failed"
    onnx.save(model_onnx, args.export_path)

    print(f"Model exported to {args.export_path}")

    # print("Exporting ONNX model to IR... This may take a few minutes.")
    # ov_model = ov.convert_model(args.export_path)
    # ov.save_model(ov_model, "xfeat.xml", compress_to_fp16=True)

    # import blobconverter

    # blobconverter.from_onnx(
    #     model="./onnx_weights/xfeat.onnx",
    #     data_type="FP16",
    #     shaves=4,
    # )
