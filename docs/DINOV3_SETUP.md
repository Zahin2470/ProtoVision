# Getting real DINOv3 weights running (do this on your Mac, not in a sandbox)

I can't do this step for you — DINOv3 weights are gated by Meta, and the
download servers aren't on any sandbox's reachable network anyway. Here's
exactly what to do, confirmed against the official repo (Aug 2026).

## 1. Clone the official repo (this part is public, no gate)

```bash
git clone https://github.com/facebookresearch/dinov3 dinov3_repo
```

Set `PROTOVISION_DINOV3_REPO` to wherever you put it, or just clone it as
`./dinov3_repo` relative to the project root (that's the default).

## 2. Request access to the weights

Go to the model table in the repo's README and click a download link for
`dinov3_vits16` (LVD-1689M / web-pretrained). This opens Meta's request-access
form. Once approved, Meta emails you a list of signed URLs — one per
checkpoint/variant.

⚠️ Use `wget` on the URL, not a browser — the repo explicitly calls this out
(the signed URLs are one-shot/streaming-friendly and browsers can mangle
large downloads).

```bash
wget "<the emailed dinov3_vits16 URL>" -O data/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

That exact filename isn't a guess — it matches the hash (`08c60483`) baked
into the `dinov3_vits16` hub entrypoint itself, so it's what `load_default_backbone()`
looks for by default. If Meta emails a differently-named file, either rename
it or point `PROTOVISION_DINOV3_WEIGHTS` at it directly — the loader also
accepts the raw emailed URL instead of a local path, if you'd rather not
store the file:

```bash
export PROTOVISION_DINOV3_WEIGHTS="<the emailed URL>"
```

## 3. License note

DINOv3's weights ship under a **custom Meta license** (not Apache/MIT like
DINOv2) — commercial use is technically allowed, but redistribution of the
weights or derivatives must carry the same license terms, and if you publish
research using it (relevant for your GreenNet-style IEEE work later, or if
you write this project up), the license requires acknowledging DINO
Materials in the publication. Full text: `https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md`.
This is not legal advice — just flagging it since your other projects are
portfolio pieces you might publish/share.

## 4. Sanity check on your machine

```bash
python -c "
from protovision.backbone import load_default_backbone
import numpy as np
bb = load_default_backbone()
img = (np.random.rand(224, 224, 3) * 255).astype('uint8')
emb = bb.embed(img)
print(emb.shape, emb.dtype)
"
```

Expect `(384,) float32`. This confirms the real weights load and produce the
right shape — it does NOT confirm semantic quality; for that, compare
embeddings of two photos of the same real object vs. two different objects
(exactly the acceptance check described in the original brief) once you have
a camera to grab crops with.

## macOS-specific things to check (I can't verify these from here)

- The DINOv3 repo says it's tested on Linux + PyTorch ≥2.7.1. Pure inference
  (no training-only ops) is very likely fine on macOS CPU, but please
  double check nothing in the ViT forward pass (e.g. its RoPE positional
  embeddings) throws on your PyTorch build before we build the live loop on
  top of it.
- Don't try the `mps` (Apple GPU) device yet — some custom ops in the model
  may not have MPS kernels. Start on `device="cpu"` (the default in
  `load_default_backbone`), confirm correctness, and only try `mps` as an
  optimization once we're at the "is it fast enough" stage.
