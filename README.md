# Anonymous AAAI-27 Submission Code

This repository contains the implementation and analysis workflow for an anonymous AAAI-27 submission on sparse positional soft-prefix updates and reusable prefix KV caches.
It was built upon Karpathy's public repository, nanoGPT.

## Scope

The experiments use frozen GPT-2 Small on WikiText-2. A learned positional soft prefix is updated once every \(m\) iterations. During frozen iterations, its layerwise key/value tensors may be cached and reused.

The principal study compares:

- positional prefix length \(L\);
- prefix update period \(m\);
- cache-on and matched cache-off execution;
- training wall time, validation perplexity, and dedicated evaluation time.

## Repository layout

```text
train.py                         Training implementation
model.py                         GPT-2 and positional soft-prefix implementation
config/h1_wikitext2.py           Principal WikiText-2 configuration
data/wikitext2/prepare.py        WikiText-2 download and tokenization
h2/h2_train_sweep.py             Cache-on positional-prefix sweep
h2/h2_train_sweep_cache_off.py   Cache-off positional-prefix sweep
h2/h2_eval_sweep_by_id_cacheonoff.py
                               
```


## Environment

Use Python 3.10+ with a CUDA-enabled PyTorch installation.

```bash
pip install -r requirements.txt
```

The main training configuration disables Weights & Biases logging by default. It can therefore be used for local training without an external account.

## Prepare WikiText-2

Ensure the right folders and W&B project names appear under /config. Then, process your dataset.

```bash
python data/wikitext2/prepare.py
```

This creates `train.bin`, `val.bin`, and `meta.pkl` in `data/wikitext2/`.

## Training

A single cache-on configuration can be run as follows:

```bash
python train.py config/h1_wikitext2.py \
  --prefix_len=64 \
  --prefix_update_period=5 \
  --prefix_cache=True \
  --max_iters=2500 \
  --learning_rate=0.1 \
  --out_dir=out_L64_m5_cache_on
```

For the matched cache-off control, change only:

```bash
--prefix_cache=False
```

The cache-on and cache-off sweep scripts are:

```bash
python h2/h2_train_sweep.py
python h2/h2_train_sweep_cache_off.py
```

Before launching a sweep, set the intended prefix lengths and update periods in the corresponding script.

## Matched evaluation

`h2/h2_eval_sweep_by_id_cacheonoff.py` evaluates selected trained prefixes under cache-on or cache-off execution using identical random seeds and validation sampling settings.

The script retrieves trained prefix artifacts through Weights & Biases. To use this artifact-backed evaluation path, configure a project under your own W&B account, populate `PROJECT` and `RUN_IDS`, and enable W&B logging when training the corresponding checkpoints. No account, project, run identifier, or URL is included in this anonymous submission package.

The dedicated evaluation settings are:

- evaluation batches: 50
- evaluation batch size: 2
- seed: 1337
- dropout: 0.0

## Paper tables and figures

The analysis notebook consumes anonymized exported result CSV files and generates the paper tables and figures. It keeps cache-on/cache-off comparisons matched at fixed \((L,m)\).

The dense prefix-length table uses the cache-off, \(m=1\) runs. The sparse quality--time grid uses cache-on runs. Matched training and dedicated evaluation tables compare cache on and cache off at the same \((L,m)\).

## Principal experimental configuration

| Parameter | Value |
|-----------|-------|
| Backbone | pretrained GPT-2 Small (124M) |
| Dataset | WikiText-2 |
| Prefix type | positional soft prefix |
| Prefix lengths | 0, 16, 32, 64, 80, 100, 256, 512, 760 |
| Update periods | 1, 5, 10, 20 |
| Training iterations | 2,500 |
| Learning rate | 0.1 |
| Micro-batch size | 6 |
| Gradient accumulation | 10 |
| Precision | FP16 |
| Seed | 1337 |
| Hardware | NVIDIA T4, 16 GB |

## Anonymity

This submission package intentionally contains no author names, affiliations, repository URLs, Weights & Biases entities, Weights & Biases project names, Weights & Biases run identifiers, API keys, Kaggle account information, local paths, Git history, or checkpoints.