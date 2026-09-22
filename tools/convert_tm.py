"""tm-my-image-model (Teachable Machine, TF.js) -> Keras -> ONNX.

Einmalig laufen lassen (braucht TensorFlow, wird NICHT für die App benötigt):

    pip install "tensorflow-cpu==2.15.0" "tensorflowjs==4.22.0" "tf2onnx==1.16.1"
    python tools/convert_tm.py --model-dir model --out tm-model.onnx

Erwartete Eingabe: model/model.json + model/weights.bin (aus tm-my-image-model.zip).
Ausgabe: tm-model.onnx mit Input NHWC float32 [1, 224, 224, 3], Output Logits [1, 2].
Labels/Reihenfolge: s. metadata.json -> ["Taschenrechner", "Federtasche"].
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--out", default="tm-model.onnx")
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    import tensorflowjs as tfjs  # type: ignore

    model_dir = Path(args.model_dir)
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    print("labels:", meta.get("labels"), "| imageSize:", meta.get("imageSize"))

    keras_path = model_dir / "tm-my-image-model.keras"
    print("1/3 TF.js -> Keras ...")
    model = tfjs.converters.load_keras_model(str(model_dir / "model.json"))
    model.save(str(keras_path))
    print("    inputs:", model.inputs[0].shape, "outputs:", model.outputs[0].shape)

    print("2/3 Keras -> ONNX ...")
    import tf2onnx  # type: ignore
    import tensorflow as tf

    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(model, input_signature=spec,
                               opset=args.opset, output_path=args.out)
    print("3/3 fertig:", args.out)


if __name__ == "__main__":
    main()
