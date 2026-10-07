# AI masks (Object and Describe)

LightCraft selects objects with **SAM 3** (Segment Anything with Concepts, Meta 2025), run in
pure Rust by `crates/segment` on [candle](https://github.com/huggingface/candle): on the GPU
through Metal on macOS, on the CPU elsewhere (for now).

## Using them

In the Masking panel (M):

- **Object**: click the thing you want; every click refines the selection. ⌥-click (Alt) a part
  to leave it out. Clicks show as green (include) and red (leave out) dots.
- **Describe**: type what to select — `sky`, `trees`, `the red car` — and press Return. Every
  instance the model finds (score above 0.5) joins the mask; if nothing matches, nothing is
  created and you are told so.
- Several things at once: `car, road` (comma-separated) selects each and merges them.
- **+ / −** under the selected mask (and the Add / Subtract / Intersect menus) combine
  selections: Describe `car`, click the mask, + ▸ Describe… `road`; − ▸ Describe… `people` takes
  the people out of a sky mask; − ▸ Object then clicks remove one object.
- **Edge** (per selection, −100…100): below 0 a crisper border, above 0 a feathered one (up to 2 %
  of the long edge, so previews and exports match).
- Hover a mask in the Masks list to see it in red on the photo; O shows the selected one all the
  time.
- **Detail:** about a second after the last click (or right after a description), the photo
  around the selection is analyzed again, zoomed in, from a sharper source (~5 s on an M4 Pro,
  in the background); the result is kept as a high-resolution patch, so small objects get 5–10×
  finer edges than one pass over the whole photo gives.

The first Object or Describe on a photo analyzes it (the image encoder runs once per photo and
look: ~3 s on an Apple M4 Pro, plus a one-time ~6 s for compiling GPU kernels and loading the
model after launch). That starts in the background as soon as you pick Object; after it, each
click takes ~35 ms and each description ~0.6 s.

The result is stored with the mask: the model's 288 × 288 mask logits over the uncropped photo
(quantized, compressed, ~10–30 KB). Renders and exports sample them at any resolution and never
need the model, so masks survive moving the library to a machine without it, and the web build
renders them. Editing the photo's look later does not recompute a mask (click again to update).

## Installing the model

The weights are not part of LightCraft: they are Meta's `facebook/sam3` checkpoint on Hugging
Face, under the SAM License, and access is gated. Once:

1. Open <https://huggingface.co/facebook/sam3>, accept the license and wait for approval.
2. Create a read token (Hugging Face ▸ Settings ▸ Access Tokens), or log in with `hf auth login`.
3. Run the installer from the repository:

```sh
HF_TOKEN=hf_... tools/install-sam3.sh            # macOS, Linux
tools/install-sam3.sh --check                     # verify an installation
```

```powershell
$env:HF_TOKEN = "hf_..."; .\tools\install-sam3.ps1   # Windows
```

It downloads `model.safetensors` (3.4 GB, fp32; resumes when interrupted, checked against the
official SHA-256), `vocab.json`, `merges.txt` and the small config files into the folder
LightCraft looks in: `~/Library/Application Support/LightCraft/models/sam3/` (macOS),
`%APPDATA%\LightCraft\models\sam3` (Windows), `~/.config/lightcraft/models/sam3` (Linux).
`--dir` (or `LIGHTCRAFT_SAM3_DIR`, which LightCraft also reads) picks another folder; `--repo`
another Hugging Face repository with the same files. Restart LightCraft afterwards.

Nothing else is needed at run time: the model runs in LightCraft itself (pure Rust, no Python).
Without the files, Object and Describe say where they are expected.

**Developers:** the accuracy test against the Python reference needs `torch` and
`transformers` (`tools/requirements-sam3-reference.txt`):

```sh
python3 -m venv .venv-sam3 && .venv-sam3/bin/pip install -r tools/requirements-sam3-reference.txt
.venv-sam3/bin/python tools/sam3_reference.py photo.jpg ref.safetensors
LIGHTCRAFT_SAM3_DIR=<model dir> LIGHTCRAFT_SAM3_REF=ref.safetensors cargo test -p lightcraft-segment --release -- --nocapture
```

## Commands (control channel, MCP)

| Command | Params | |
|---|---|---|
| `mask.add` | `{kind: "object", points?: [[x,y],…], exclude?: [[x,y],…]}` | an Object mask (empty until clicked) |
| `mask.add` | `{kind: "prompt", text}` | a Describe mask |
| `mask.addComponent` | `{op: add\|subtract\|intersect, kind: "object"\|"prompt", …}` | the same as a component |
| `mask.objectPoint` | `{x, y, exclude?: bool, id?}` | one click on the selected mask's Object selection |
| `mask.refineDetail` | `{id?, component?}` | the zoomed-in detail pass, in the background → `{started}` |
| `segment.prepare` | `{}` | analyze the active photo in the background → `{busy}` |

Coordinates are normalized to the uncropped, oriented photo, like every mask shape.

## Implementation notes

- `crates/segment` ports the Hugging Face `transformers` implementation (Apache-2.0; see
  `NOTICE`): the 32-layer ViT backbone with 2-D RoPE and windowed attention, the feature
  pyramids, the SAM 2-style prompt encoder and two-way mask decoder for clicks, and the CLIP text
  encoder, DETR encoder/decoder (box relative-position bias, presence token) and pixel decoder
  for text. The CLIP tokenizer is a small BPE in `tokenizer.rs`.
- candle is pinned at 0.9.2: later releases make `candle-core` depend on `tokenizers` with the
  Oniguruma C library, and the product is pure Rust. Weights are read with positional reads (no
  memory map: `unsafe` stays in `lightcraft-sysmem`), and only the tensors a path needs.
- Accuracy: `crates/segment/tests/reference.rs` compares every stage against the Python
  reference (`tools/sam3_reference.py`); on Metal the backbone matches to 7e-5 relative error
  and masks to ~5e-5. f16 was tried and rejected (overflow, ~7 % faster).
- The engine feature `sam` (on in the desktop app, off for web and CLI) gates the model; without
  it the commands report that the build has no AI masks.
