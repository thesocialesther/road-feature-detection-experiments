#!/usr/bin/env python3
"""Export the uploaded YOLO11 checkpoint and check PyTorch/ONNX parity."""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import tempfile


def run(args):
    # Keep Ultralytics settings in a writable temporary directory.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "bumblebee-yolo-config"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import ultralytics
    from ultralytics import YOLO

    source = args.weights.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; choose a different --output to preserve it")
    if destination.suffix != ".onnx":
        raise ValueError("--output must end in .onnx")
    expected_names = ["cracks", "speed_bumps", "potholes", "left_road_boundaries", "right_road_boundaries"]
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="yolo11-export-") as directory:
        staged_weights = Path(directory) / "yolo11n.pt"
        shutil.copy2(source, staged_weights)
        model = YOLO(str(staged_weights), task="detect")
        names = [model.names[i] for i in range(len(model.names))]
        if names != expected_names:
            raise ValueError(f"Expected the five trained road classes, received {names}")
        model.model.cpu().float().eval()
        rng = np.random.default_rng(42)
        inputs = [np.full((1, 3, args.imgsz, args.imgsz), .5, np.float32),
                  rng.random((1, 3, args.imgsz, args.imgsz), dtype=np.float32)]
        reference = []
        with torch.inference_mode():
            for tensor in inputs:
                output = model.model(torch.from_numpy(tensor))
                if isinstance(output, (tuple, list)):
                    output = output[0]
                reference.append(output.detach().cpu().numpy())
        exported = model.export(format="onnx", imgsz=args.imgsz, batch=1,
                                opset=17, simplify=False, dynamic=False,
                                half=False, nms=False, device="cpu")
        graph = onnx.load(str(exported))
        onnx.checker.check_model(graph, full_check=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session = ort.InferenceSession(str(exported), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        metadata_names = ast.literal_eval(session.get_modelmeta().custom_metadata_map["names"])
        if [metadata_names[i] for i in range(len(names))] != names:
            raise ValueError("Exported class metadata differs from the checkpoint")
        comparisons = []
        for tensor, expected in zip(inputs, reference):
            actual = session.run(None, {session.get_inputs()[0].name: tensor})[0]
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise ValueError("Invalid exported output shape or non-finite predictions")
            np.testing.assert_allclose(actual[:, :4], expected[:, :4], rtol=1e-3, atol=1e-2)
            np.testing.assert_allclose(actual[:, 4:], expected[:, 4:], rtol=1e-3, atol=1e-4)
            comparisons.append({"max_box_error": float(np.abs(actual[:, :4] - expected[:, :4]).max()),
                                "max_score_error": float(np.abs(actual[:, 4:] - expected[:, 4:]).max())})
        report = {"checkpoint": str(source), "ultralytics": ultralytics.__version__,
                  "input_shape": session.get_inputs()[0].shape,
                  "output_shape": session.get_outputs()[0].shape, "classes": names,
                  "onnx_check": "passed", "parity": comparisons}
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Publish only after graph and numerical checks pass.
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".onnx", delete=False) as file:
            temporary = Path(file.name)
        try:
            shutil.copyfile(exported, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        destination.with_suffix(".export.json").write_text(json.dumps(report, indent=2))
        print(f"Validated model saved to {destination}")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path(__file__).with_name("yolo11n.pt"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("yolo11n.onnx"))
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    if args.imgsz < 32 or args.imgsz % 32:
        parser.error("--imgsz must be a positive multiple of 32")
    run(args)
