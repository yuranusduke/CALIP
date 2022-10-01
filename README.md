# CALIP

### Unofficial implementation of *[CALIP: Zero-Shot Enhancement of CLIP with Parameter-free Attention](https://arxiv.org/abs/2209.14169)*


## Acknowledgement
Best thanks to paper's authors for great contribution and also to [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch) and [CoOP](https://github.com/KaiyangZhou/CoOp),
this repo is produced based on above gorgeous codes.
Thus, you may have to install Dassl before going further.


## Run

### Few-Shot Learning

In this scenario, we need to use CALIP\_FS, which means model has to be fine-tuned due to introduction of linear layers in attention.

All you need is `CALIP/scripts/fewshots.sh`, which contains two input arguments.

`DATASET` takes as input a dataset name, like `imagenet` or `caltech101`. The valid names are the files' names in `CALIP/configs/datasets/`.

We fix `CFG` to be `rn50_ep200_ctxv1.yaml` in order to follow original setup in paper for training CALIP\_FS.
Below we provide examples on how to run CALIP\_FS on OxfordsPets.

- 1 shot: `bash scripts/fewshots.sh oxford_pets 1`
- 2 shots: `bash scripts/fewshots.sh oxford_pets 2`
- 4 shots: `bash scripts/fewshots.sh oxford_pets 4`
- 8 shots: `bash scripts/fewshots.sh oxford_pets 8`
- 16 shots: `bash scripts/fewshots.sh oxford_pets 16`

### Zero-Shot CLIP

See `CALIP/scripts/zeroshot.sh` for using CALIP\_PF(CALIP with parameter-free attention).

`bash scripts/zeroshot.sh oxford_pets`

### Modify Settings

One can also modify settings in `main.py` `extend_cfg` function, where you can specify optimal hyper-parameters denoted in the paper. And according to original paper, both max and avg pooling are utilized for obtaining $F_v^{a}$, but there is ambiguity that how to combine them, in implementation, we only add them together, results may be influcenced by this operation.