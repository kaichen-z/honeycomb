<div align="center">

  <h1>Honeycomb: Constant-Size Scene Memory<br/>Representation for Video World Models</h1>

  <p>
    Jack Wei Lun Shi<sup>2,*</sup> &nbsp;
    Kaichen Zhou<sup>1,3,*</sup><br/>
    Haoyu Chen<sup>1</sup> &nbsp;
    Yufeng Weng<sup>2</sup> &nbsp;
    Keane Ong<sup>2,3</sup> &nbsp;
    Ruojin Cai<sup>1</sup> &nbsp;
    Hang Hua<sup>4</sup><br/>
    Justin K.W. Yeoh<sup>2</sup> &nbsp;
    Mengyu Wang<sup>1</sup>
  </p>

  <p>
    <sup>1</sup>Harvard University &nbsp;&nbsp;
    <sup>2</sup>National University of Singapore<br/>
    <sup>3</sup>MIT &nbsp;&nbsp;
    <sup>4</sup>MIT-IBM Watson AI Lab
  </p>

  <p><sup>*</sup>Equal contribution.</p>

</div>

We introduce Honeycomb, a video world model built on HexMemory, our proposed low-rank representation for storing scene features in a fixed-size memory with a total of six spatial and spatiotemporal planes.

To prepare data and train the model, follow the numbered sections: prepare clips → train the writer → build the memory-conditioned dataset → train stages 1 and 2. If you already have trained checkpoints and a prepared input clip, go to [Inference](#inference).

## Code layout

| Directory | Purpose |
| --- | --- |
| `honeycomb/` | Memory bounds, latent handling, and readout utilities |
| `shared_writer/` | Shared writer architecture, splatting, and training helpers |
| `recurrent_writer/` | Recurrent memory updates and writer training (default) |
| `replacement_writer/` | Replacement writer and rollout-pack preparation shared by both writers |
| `adapter/` | Corpus building and the HexMemory rollout adapter |
| `data_process/` | Clip preparation, geometry, captions, VAE encoding, and LMDB packing |
| `scripts/` | Corpus-building, training, and inference entry points |
| `src/lsm/` | Video backbone, conditioning modules, and training utilities |
| `src/lsm/inference/` | Generation pipeline, chunk scheduling, and depth worker |

## Environment and pretrained weights

Run all commands from this directory. Replace paths beginning with `/PATH-TO-` with your own locations.

```bash
conda env create -f environment.yml
conda activate honeycomb
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Place the pretrained Wan2.2-TI2V-5B files under `data/Wan-AI/Wan2.2-TI2V-5B/`:

```text
diffusion_pytorch_model*.safetensors
Wan2.2_VAE.pth
models_t5_umt5-xxl-enc-bf16.pth
google/umt5-xxl/
```

## 1. Prepare clips and writer packs

The writers train jointly on RealEstate10K and SpatialVID. The video model's two training stages use the RealEstate10K corpus produced by the recurrent writer.

The [data preparation guide](data_process/README.md) describes clip processing for Honeycomb. Use these preparation settings before running collection and sample construction:

```bash
export HONEYCOMB_MAX_VIDEOS=none
export HONEYCOMB_REF_CANDIDATE_SCOPE=past_only
export HONEYCOMB_APPLY_DYNAMIC_MASK=0
```

Set `HONEYCOMB_VIDEO_DIRS` and `HONEYCOMB_OUTPUT_ROOT` for each dataset. Use `HONEYCOMB_EPS_IOU=0.04` for RealEstate10K and `HONEYCOMB_EPS_IOU=0.01` for SpatialVID. Generate captions for both `clip` and `train_target_rgb`.

The examples below use prepared clip directories `data/re10k/train/` and `data/spatialvid/train/`. Encode the full clip as well as the training views; the writers require `clip.pt` alongside `geometry.npz`:

```bash
for corpus in re10k spatialvid; do
  python -m data_process.run_video_vae_encode \
    --input-root "data/$corpus/train" \
    --video-keys clip,train_preceding_rgb,train_target_rgb,train_reference_rgb \
    --skip-existing
done
```

Create separate lists of clips for writer training and validation. Save them as `data/splits/re10k_train.json`, `re10k_val.json`, `spatialvid_train.json`, and `spatialvid_val.json`. Each file contains clip-folder names as strings, such as `["00000000", "00000001"]`.

Create the writer's training files, called rollout packs. Both writer variants use these files:

```bash
for corpus in re10k spatialvid; do
  for split in train val; do
    OMP_NUM_THREADS=1 python replacement_writer/prep_pack_rollout.py \
      --src "data/$corpus/train" \
      --out "data/$corpus/packs_$split" \
      --ids "data/splits/${corpus}_${split}.json"
  done
done
```

## 2. Train the writers

The Python defaults contain the training settings: 500,000 steps, learning rate `5e-4`, and plane-pair ranks `(48, 48, 48)`.

```bash
TRAIN_PACKS="data/re10k/packs_train,data/spatialvid/packs_train"
VAL_PACKS="data/re10k/packs_val,data/spatialvid/packs_val"

python recurrent_writer/train_rec.py \
  --packs "$TRAIN_PACKS" --val-packs "$VAL_PACKS" \
  --out runs/recurrent_writer --tag recurrent
```

The recurrent writer is our default model. The replacement writer is an optional comparison model; you do not need to train it to continue. To train it:

```bash
python replacement_writer/train_rep.py \
  --packs "$TRAIN_PACKS" --val-packs "$VAL_PACKS" \
  --out runs/replacement_writer --tag replacement
```

The recurrent writer saves `runs/recurrent_writer/recurrent.ckpt`. Use the completed 500,000-step checkpoint for the corpus-building step below.

## 3. Build the HexMemory corpus

Build the HexMemory training corpus with the trained writer. This script saves the memory latents alongside the prepared RealEstate10K clips and packs the dataset into LMDB:

```bash
SRC="$PWD/data/re10k/train" \
OUT="$PWD/data/re10k_hexmemory/train" \
WRITER="$PWD/runs/recurrent_writer/recurrent.ckpt" \
bash scripts/build-hexmemory-corpus.sh
```

The script uses GPUs `0 1 2 3 4 5 6 7` by default; set `GPUS` to change the worker devices. The LMDB output is `data/re10k_hexmemory/train_lmdb/`.

## 4. Train the video model

Stage 1 trains the conditioning branch that connects HexMemory to the video model. Stage 2 freezes that branch and trains LoRA adapters on the video model.

Both launchers use eight GPUs, BF16, batch size 1 per GPU, and eight gradient-accumulation steps, giving an effective batch size of 64. Keep the per-GPU batch size at 1 because samples have variable numbers of conditioning frames.

| Setting | Stage 1 | Stage 2 |
| --- | --- | --- |
| Trainable components | VACE conditioning branch | Backbone LoRA; VACE frozen |
| Steps | 10,000 | 5,000 |
| Learning rate | `1e-5` | `1e-4` |
| Schedule | Cosine to zero | Cosine to zero |
| Preceding-conditioning noise maximum timestep | 0 | 50 |
| LoRA rank / alpha | — | 64 / 64 |

The conditioning branch uses eight Wan-initialized blocks at stride 4, a randomly initialized patch embedding, zero-initialized output projections, and 49 input channels. Both stages use reference–preceding–target ordering and text-prompt dropout of 0.2.

```bash
export MODEL_ROOT="$PWD/data/Wan-AI/Wan2.2-TI2V-5B"
export DATA_PATH="$PWD/data/re10k_hexmemory/train_lmdb"

OUTPUT_DIR="$PWD/runs/stage1" bash scripts/train-stage1.sh

OUTPUT_DIR="$PWD/runs/stage2" \
VACE_CKPT="$PWD/runs/stage1/vace/step_0010000_vace.safetensors" \
bash scripts/train-stage2.sh
```

The final checkpoints are `runs/stage1/vace/step_0010000_vace.safetensors` and `runs/stage2/lora/step_0005000_lora.safetensors`.

## Inference

You need a prepared clip, the pretrained Wan weights, and all three trained checkpoints: the writer, stage-1 VACE, and stage-2 LoRA. The command reads the text description from `clip.txt` and camera/depth information from `geometry.npz`.

Depth estimation uses ViPE's Depth Anything 3 installation. Set `--depth-python` to the Python executable in that environment.

```bash
MODEL_ROOT="$PWD/data/Wan-AI/Wan2.2-TI2V-5B"
python scripts/honeycomb_inference.py \
  --geometry-path /PATH-TO-CLIP/geometry.npz \
  --prompt-path /PATH-TO-CLIP/clip.txt \
  --output-dir runs/inference \
  --model-config "$MODEL_ROOT/diffusion_pytorch_model*.safetensors" \
  --model-config "$MODEL_ROOT/Wan2.2_VAE.pth" \
  --model-config "$MODEL_ROOT/models_t5_umt5-xxl-enc-bf16.pth" \
  --tokenizer-path "$MODEL_ROOT/google/umt5-xxl" \
  --writer-checkpoint runs/recurrent_writer/recurrent.ckpt \
  --vace-checkpoint runs/stage1/vace/step_0010000_vace.safetensors \
  --lora-checkpoint runs/stage2/lora/step_0005000_lora.safetensors \
  --depth-python /PATH-TO-VIPE/bin/python \
  --num-frames 81
```

Keep `clip.mp4` beside `geometry.npz` when RGB frames are not embedded in the geometry file. The input must provide camera poses for the requested trajectory.

The model generates the video in chunks. Before continuing, it encodes the last generated frame (the anchor), the preceding frames (P), and selected older reference frames (R) back into the model's latent representation. It uses 40 denoising steps per chunk and updates HexMemory between chunks. LoRA is applied from the second chunk onward.

The finished video is saved to `runs/inference/videos/generated.mp4`. The same output directory contains `metadata.json` with the run settings and `hexmemory.json` with the memory-update records.

## Acknowledgements and license

This implementation builds on Latent Spatial Memory, Spatia, and Wan. See [LICENSE](LICENSE) for the repository license.
