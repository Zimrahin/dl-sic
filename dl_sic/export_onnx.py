import argparse
import os
import re
import torch
from model.real_tdcr_net import RealTDCRNet


def main():
    parser = argparse.ArgumentParser(description="Export RealTDCRNet to ONNX")
    parser.add_argument(
        "-w", "--weights_path", type=str, help="Path to model weights (.pth file)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output.onnx", help="Output ONNX file path"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        choices=[11, 12, 13, 14, 15],
        help="ONNX opset version",
    )
    parser.add_argument("--M", type=int, default=128, help="Middle channels in encoder")
    parser.add_argument(
        "--N", type=int, default=64, help="Output channels of encoder and LSTM input"
    )
    parser.add_argument(
        "--U", type=int, default=128, help="Middle channels in dilated convolution"
    )
    parser.add_argument(
        "--V", type=int, default=8, help="Dilated convolutions on each side of LSTM"
    )
    args = parser.parse_args()

    assert args.weights_path, "No model directories found. Specify --weights"
    weights_path = args.weights_path

    M, N, U, V = args.M, args.N, args.U, args.V
    print(f"Loading model: M={M}, N={N}, U={U}, V={V}")
    print(f"Weights: {weights_path}")

    model = RealTDCRNet(M=M, N=N, U=U, V=V, dtype=torch.float32)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # (batch, channels, T)
    input = torch.rand((1, 2, 10000), dtype=torch.float32)

    torch.onnx.export(
        model,
        (input,),
        args.output,
        export_params=True,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "sequence_length"},
            "output": {2: "sequence_length"},
        },
    )
    print(f"ONNX model exported to {args.output}")


if __name__ == "__main__":
    main()
