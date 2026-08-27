# Third-party baseline adapters

The baseline backbones are upstream repositories, used at the commits below.
Only the files we added or changed are kept here; clone the upstream repo, copy
the adapter files to the same relative paths, then apply the patch.

| Baseline | Upstream | Commit used |
|---|---|---|
| AMD (encoder base) | https://github.com/TROUBADOUR000/AMD | see note below |
| PatchTST | https://github.com/yuqinie98/PatchTST | `204c21e` |
| TimeMixer | https://github.com/kwuking/TimeMixer | `e246105` |
| iTransformer | https://github.com/thuml/iTransformer | `c2426e6` |
| START (appendix only) | https://github.com/aptx1231/START | `8e5dfca` (unmodified) |
| TrajMamba (appendix only) | https://github.com/yichenliuzong/TrajMamba | `4e25ede` (unmodified) |
| HGT-RN (appendix only) | https://github.com/zjy9826/HGT-RN | `513f819` (unmodified) |

```bash
git clone https://github.com/kwuking/TimeMixer && cd TimeMixer && git checkout e246105
cp -r ../third_party_adapters/TimeMixer/{main_traj.py,utils} .
git apply ../third_party_adapters/TimeMixer/patches/upstream_changes.patch
```

The same procedure applies to PatchTST (`PatchTST_supervised/main_traj.py`, no
patch) and iTransformer (`main_traj.py`, `run_porto.py`,
`data_provider/trajectory_loader.py`, `utils/geo_metrics.py`,
`utils/traj_dataloader.py`, plus its patch).

Every adapter reuses the shared trajectory dataloader and the shared meter-level
ADE/FDE evaluator, so all baselines see identical inputs, splits and metrics.
The appendix-only repositories (START, TrajMamba, HGT-RN) were run unmodified;
their GPS-only adapted encoder + forecasting-head variants are implemented in
`benchmarks/scripts/run_extended_sota_adapters.py`.

## Encoder provenance

PriMoTraj's temporal encoder is built on the released implementation of AMD
(Hu et al., *Adaptive Multi-Scale Decomposition Framework for Time Series
Forecasting*, AAAI 2025). `primotraj/models/common.py` (Parallel Mixer, PM;
Multi-scale Decomposable Mixing, MDM; RevIN) and `primotraj/models/tsmoe.py`
(Adaptive Multi-predictor Synthesis, AMS) derive from that code base, and the
file name `tsAMD.py` and the `AMD` alias at the end of that module are kept for
backward compatibility with existing checkpoints and scripts.

The contributions of this paper -- the deterministic motion-prior bank, the
sample-wise gate over it, and the bounded residual head -- are in
`primotraj/models/tsAMD.py`. The AMS mixture head is disabled in the deployed
configuration (`--use_moe False`).

Please check the upstream AMD licence before redistributing these files.
