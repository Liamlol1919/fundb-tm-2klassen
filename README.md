# fundb-tm-2klassen 🎒🔢✏️

Fundbüro-Prototyp (Katharineum zu Lübeck) — Umbau von
[`Liamlol1919/Fundb-roprototypkochjugit`](https://github.com/Liamlol1919/Fundb-roprototypkochjugit),
aber mit **deinem Teachable-Machine-Modell `tm-my-image-model`** statt ImageNet.

Erkennt **nur 2 Kategorien** (so gewollt):

| # | Kategorie | Icon |
|---|-----------|------|
| 1 | Taschenrechner | 🔢 |
| 2 | Federtasche | ✏️ |

Alles unter 50 % Konfidenz → `Sonstiges` (Fallback).

## Modell-Dateien

| Datei | Herkunft |
|-------|----------|
| `model/model.json` + `model/weights.bin` + `model/metadata.json` | Original-Export aus `tm-my-image-model.zip` (Teachable Machine, TF.js, imageSize 224) — unverändert |
| `tm-model.onnx` | Konvertiert: TF.js → Keras → ONNX (s. `tools/convert_tm.py`) |
| `labels_tm.txt` | `["Taschenrechner", "Federtasche"]` (aus `metadata.json`) |

Preprocessing exakt wie Teachable Machine: `224×224`, float32, `(px / 127.5) - 1`, NHWC.

## Lokal starten

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Konvertierung reproduzieren

```bash
pip install "tensorflow-cpu==2.15.0" "tensorflowjs==4.22.0" "tf2onnx==1.16.1"
python tools/convert_tm.py --model-dir model --out tm-model.onnx
```

## Deploy (Streamlit Cloud)

1. Repo verbinden, Branch `main`, File `app.py`, Python `3.11` (`runtime.txt`).
2. Erster Upload lädt das ONNX-Modell (~10 s), danach ist die Erkennung schnell.

## Unterschiede zum Original-Repo

- `VISION_CLASS_TO_CATEGORY` (ImageNet → 8 Kategorien) **entfernt**.
- `load_vision_model()` lädt nur noch `tm-model.onnx` via `onnxruntime`.
- `analyze_image_ai()` mit TM-Normierung, Softmax, 50-%-Schwelle.
- `CATEGORIES` / `KAT_ICON` / Demo-Items auf die 2 Klassen reduziert.
- `requirements.txt` unverändert (kein TensorFlow nötig — nur `onnxruntime`).
