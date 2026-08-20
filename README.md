<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,25:1e3a8a,55:2563eb,80:6366f1,100:8b5cf6&height=140&section=header&text=🔬%20Proto+Vision&fontSize=64&fontAlignY=38&animation=fadeIn&fontColor=ffffff" width="100%" alt="ProtoVision"/>

<h2>🔬 Self-Supervised Few-Shot Object Recognition</h2>

</div>

<br>

<p>
  <img src="https://img.shields.io/badge/🧠%20Backbone-DINOv3%20ViT--S%2F16-0f172a?style=for-the-badge">
  <img src="https://img.shields.io/badge/🔢%20Embedding-384%20Dim-2563eb?style=for-the-badge">
  <img src="https://img.shields.io/badge/🎯%20Similarity-Cosine-7c3aed?style=for-the-badge">
  <img src="https://img.shields.io/badge/🔬%20Learning-Few--Shot-f97316?style=for-the-badge">
</p>

<br>

<blockquote>
  <strong>Show a few examples.</strong> Build a prototype. <strong>Recognize unseen objects instantly.</strong>
</blockquote>

</div>

---

<div align="center">

### 🧩 The Core Idea

`Examples` → `DINOv3` → `Embeddings` → `Prototype` → `Cosine Similarity` → `Prediction`

</div>

---

## 🧠 What is ProtoVision?

**ProtoVision** is a few-shot object recognition pipeline built around a **frozen DINOv3 ViT-S/16 backbone**.

Instead of training a classifier for every new object, ProtoVision:

1. Takes a handful of example images for an object.
2. Converts each image into a DINOv3 embedding.
3. Stores those embeddings as a **class prototype**.
4. Compares new observations using **cosine similarity**.
5. Can return an **"unknown"** result when similarity falls below the configured threshold.

The core idea is simple:

```text
Example Images
      │
      ▼
┌─────────────────┐
│ Frozen DINOv3   │
│   ViT-S / 16    │
└────────┬────────┘
         │
         ▼
   384-d Embeddings
         │
         ▼
┌─────────────────┐
│ Class Prototype │
│   Storage       │
└────────┬────────┘
         │
         ▼
 New Image / Frame
         │
         ▼
┌─────────────────┐
│ Cosine Similarity│
│ mean / max mode │
└────────┬────────┘
         │
         ▼
   Class / Unknown
```

---

## ✨ Project Highlights

| Capability | Description |
|---|---|
| 🧊 Frozen backbone | Uses DINOv3 without a training or fine-tuning loop |
| 🎯 Few-shot recognition | Learns a class from a handful of example images |
| 🧬 Prototype-based | Keeps example embeddings instead of requiring a trained classifier |
| 🔎 Flexible matching | Supports both `mean` and `max` matching modes |
| 🚫 Open-set fallback | Can return `unknown` when similarity is below the threshold |
| 💾 Persistent prototypes | Prototypes can be saved to and loaded from JSON |
| 🧪 Strong test coverage | Current test suite reports **172/172 passing** |
| 🖥️ CPU-oriented starting point | Designed to work without requiring a GPU for the tested pipeline |

---

## 📊 Current Development Status

> **Phase 1 — all 4 steps complete and tested**

| Step | Component | Status |
|:---:|---|:---:|
| **1** | `backbone.py` — frozen DINOv3 loading + embedding extraction | ✅ Complete |
| **2** | `prototypes.py` — prototype storage + `best_match()` | ✅ Complete |
| **3** | `capture.py` / `enroll.py` / `live.py` — camera applications | ✅ Logic complete & tested · camera loop itself unverified (needs real hardware) |
| **4** | CLI — `main.py enroll` / `main.py live` / `main.py list` | ✅ Complete & tested · real enroll/live runs need real hardware + weights |

---

## 🏗️ Architecture

```mermaid
flowchart LR
    A["Example Images"] --> B["DINOv3 ViT-S/16"]
    B --> C["384-d CLS Embedding"]
    C --> D["Prototype Store"]

    E["New Image / Frame"] --> B
    D --> F["Cosine Similarity"]
    C --> F

    F --> G{"Threshold Check"}
    G -->|"Match"| H["Predicted Class"]
    G -->|"Below threshold"| I["Unknown"]
```

### Core modules

```text
ProtoVision/
├── main.py             # CLI entry point (enroll / live / list)
├── protovision/
│   ├── backbone.py     # DINOv3 loading + embedding extraction
│   ├── prototypes.py   # Prototype storage + similarity matching
│   ├── capture.py      # Camera wrapper + guide-box geometry/crop
│   ├── enroll.py       # Object enrollment (capture → embed → save prototype)
│   └── live.py         # Live recognition (frame-skip inference + best_match)
├── tests/              # Automated tests (172, all passing)
└── docs/
    └── DINOV3_SETUP.md
```

---

## 🖥️ Usage (CLI)

```bash
# Enroll a new object class — SPACE to capture, u to undo, Enter to finish
# early (once you've hit --min-examples), Esc to cancel
python main.py enroll --label mug

# with options
python main.py enroll --label mug --target-examples 10 --min-examples 6 --box-fraction 0.6

# Live recognition against everything enrolled so far — q or Esc to quit
python main.py live

# with options
python main.py live --threshold 0.6 --match-mode max --frame-skip 3

# See what's enrolled — reads the store only, no camera or backbone needed
python main.py list
```

`enroll`/`live` both need the real DINOv3 repo + weights in place first (see
`docs/DINOV3_SETUP.md`) — `list` doesn't, it just reads `data/prototypes.json`.
Every flag has a `--help`: `python main.py enroll --help`.

## 🧪 What Is Actually Tested?

ProtoVision separates the model-loading layer from the rest of the recognition logic so the core pipeline can be tested independently.

### ✅ Tested right now

**`prototypes.py`**

- Pure vector math + JSON I/O
- Adding examples
- Computing centroids
- `best_match()` in both `mean` and `max` modes
- Open-set `unknown` fallback
- Threshold edge cases
- Save/load round trips
- Atomic writes

**`backbone.py` pipeline**

- Resize-to-multiple-of-16 preprocessing
- ImageNet normalization
- BGR → RGB handling
- L2 normalization
- Output shape and dtype contract
- Batch embedding
- Gradient-free inference
- Compatibility with the expected DINOv3 interface:
  `forward_features(x)["x_norm_clstoken"]`
- Similarity sanity check between same-object and different-object crops

**`capture.py` geometry**

- Guide box centering, sizing (fraction of the shorter frame dimension),
  and clamping so it always fits inside the frame — including tiny/odd
  frame sizes
- Cropping math (correct region pulled, out-of-bounds boxes raise instead
  of silently corrupting)
- Overlay drawing never mutates the source frame

**`enroll.py` / `live.py` state machines**

- Full capture → embed → auto-finish-at-target flow, with `min_examples`
  enforced before a prototype can be saved
- Undo, cancel, and key-dispatch behavior (`handle_key`) in every state
- `live.py`'s frame-skip strategy specifically: verified with a call-counting
  backbone that inference only re-runs every `frame_skip`-th frame and the
  prediction is correctly held in between
- Real `__init__` logic (label stripping, default values, injected vs.
  auto-opened camera) exercised end-to-end with a fake in-memory camera
  standing in for hardware

**`main.py` CLI**

- Argument parsing: defaults, overrides, required flags, `choices=` validation
  (e.g. rejecting an invalid `--match-mode` or `--device`), and that shared
  flags (`--store`, `--device`, ...) work whether given before or after the
  subcommand
- `list` end-to-end (it needs neither a camera nor a backbone, so nothing's
  mocked there — it's tested exactly as it runs for real)
- `enroll`/`live` command wiring: correct arguments passed through to
  `EnrollApp`/`LiveApp`, success vs. cancelled exit codes, the "no classes
  enrolled yet" warning, and a clean `exit(1)` with a readable message
  instead of a raw traceback when DINOv3 isn't set up yet — all via a fake
  backbone/app, since the real ones need actual hardware

### ⚠️ Not yet validated in this environment

The following require the real DINOv3 weights and/or actual hardware:

- Real DINOv3 semantic quality on your own photos
- Camera FPS and inference latency on your Mac
- Practical comfort/size of the on-screen guide box
- The actual `run()` camera loops in `enroll.py`/`live.py` — the OpenCV
  window, live keypresses, and real-time overlay rendering. Every piece of
  *logic* those loops call (`handle_key`, `capture_example`,
  `process_frame`, `render_preview`, ...) is tested; the thin loop that
  wires them to a real window and a real camera isn't, and can't be from
  here.
- Running `python main.py enroll`/`python main.py live` for real — the CLI's
  own argument parsing and dispatch logic is tested (see above), but an
  actual end-to-end run needs the real backbone and a webcam, neither of
  which exist in this sandbox.

The current mock-model tests verify the **pipeline plumbing**, not the real-world semantic quality of DINOv3.

---

## 🧰 Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install the real DINOv3 model

Follow:

```text
docs/DINOV3_SETUP.md
```

The DINOv3 weights are gated and need to be obtained manually on your machine.

---

## 🧪 Run the Test Suite

```bash
pytest tests/ -v
```

Current result:

```text
172 tests
172 passed
0 failed
```

No camera, GPU, or DINOv3 weights are required for the current automated test suite.

---

## 🔬 Design Decisions

### 1. Why `dinov3_vits16`?

The selected backbone is:

- **DINOv3 ViT-S/16**
- **21M parameters**
- **384-dimensional CLS token**
- Patch size: **16 × 16**

The project intentionally uses the confirmed `dinov3_vits16` model rather than silently substituting DINOv2.

### 2. Why store every example?

Prototype storage keeps **every example embedding**, rather than only a running mean.

That makes it possible to support:

- `mean` matching
- `max` / k-NN-style matching
- Future debugging such as identifying which stored example produced the strongest match

### 3. CPU frame-rate strategy

The planned live pipeline does **not** assume that the backbone must run on every camera frame.

A proposed approach is:

```text
Frame 1 ──► DINOv3 ──► Prediction A
Frame 2 ─────────────────► Hold A
Frame 3 ─────────────────► Hold A
Frame 4 ─────────────────► Hold A
Frame 5 ──► DINOv3 ──► Prediction B
```

A frame-skip value such as every 5th frame is only a starting point. Real inference latency will be measured on the target machine before tuning the final value.

### 4. Testing camera-dependent code without a camera

`enroll.py` and `live.py` each open a real `Camera` as the very last step of
`__init__`. To keep them testable anyway:

- Every method containing actual *logic* (key handling, capture bookkeeping,
  frame-skip inference, guide-box rendering) only touches `self`'s plain
  attributes, never the camera directly — so it can be tested by
  constructing the object via `__new__` (skipping `__init__` entirely) and
  setting just those attributes.
- `__init__`'s own logic (argument validation, label stripping, defaults,
  camera injection) is tested separately by monkeypatching `Camera` with a
  no-op `FakeCamera` and going through the real constructor.
- Only the actual `run()` loop — the part that opens an OpenCV window and
  blocks on live keypresses — is left untested, since that genuinely needs
  a display and a webcam.

### 5. Testing the CLI without a camera or real weights

`main.py`'s `enroll`/`live` commands ultimately call `load_default_backbone()`
and construct a real `EnrollApp`/`LiveApp`, both of which need hardware this
sandbox doesn't have. So the CLI is tested one layer up: `load_default_backbone`,
`EnrollApp`, and `LiveApp` are monkeypatched with fakes for the dispatch tests
(`cmd_enroll`/`cmd_live`), which verifies argument wiring, exit codes, and the
"no classes enrolled" warning without touching hardware. `cmd_list` needs
neither a camera nor a backbone at all, so it's tested exactly as it runs for
real. Argument *parsing* (`build_parser()`) is tested directly with no mocking
needed, since it's pure `argparse` logic.

---

## 🗺️ Roadmap

- [x] Build frozen DINOv3 embedding pipeline
- [x] Implement prototype storage
- [x] Implement similarity matching
- [x] Add open-set `unknown` fallback
- [x] Add automated tests
- [x] Build camera capture flow (guide-box geometry, cropping, overlay)
- [x] Build object enrollment flow (capture/undo/finish/cancel state machine)
- [x] Build live recognition flow (frame-skip inference)
- [x] Add CLI commands (`enroll` / `live` / `list`)
- [ ] Measure real CPU inference latency
- [ ] Tune frame-skipping strategy

---

## 📁 Documentation

The current project includes:

- `docs/DINOV3_SETUP.md` — instructions for obtaining and configuring the real DINOv3 weights

---

## ⚠️ Current Limitations

**Phase 1 is complete.** All four steps — backbone, prototype storage,
camera-app logic, and CLI — are built and unit tested (172/172 passing).

What hasn't happened yet, and can't happen from this sandbox:

- The real DINOv3 repo + weights haven't been loaded and run for real (gated,
  requires your machine — see `docs/DINOV3_SETUP.md`)
- `python main.py enroll` / `python main.py live` haven't been run
  end-to-end against a real webcam
- No real-world semantic-quality check yet (same object → higher similarity
  than a different object, on actual photos rather than the mock model)

Once those checks pass on your machine, Phase 1 is genuinely done and Phase 2
(visual design system — Poppins typography, glass-panel HUD, the live
similarity-meter HUD, theme switching, audio) is next, per the original brief.

---

## 🔑 Core Idea

> **Don't train a new classifier for every object.**
>
> **Represent the object, store its examples, and compare future observations against those representations.**

---

<div align="center">

### ProtoVision

**Frozen backbone · Few-shot prototypes · Similarity-based recognition**

</div>
