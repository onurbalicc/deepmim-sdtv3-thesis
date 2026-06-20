# Run Index

This file tracks the renamed experiment folders and keeps the old folder names for reference.

## Rename Map

| No. | Old folder name | New folder name |
|---|---|---|
| 01 | `output_full_finetune_3gpu` | `01_old_full_finetune_random_3gpu_50ep_head_mismatch` |
| 02 | `output_pretrain_full_2gpu` | `02_baseline_pretrain_mae_2gpu_400ep` |
| 03 | `output_baseline_finetune_ckpt395_100ep_2gpu` | `03_baseline_finetune_ckpt395_2gpu_100ep` |
| 04 | `output_updated_stdv3_pretrain_400ep` | `04_updated_stdv3_pretrain_deepmim_logpe_400ep` |
| 05 | `output_updated_stdv3_finetune_ep75` | `05_updated_stdv3_finetune_ckpt75_3gpu_50ep` |
| 06 | `output_full_finetune_3gpu_clean_100cls` | `06_clean_baseline_full_finetune_random_3gpu_50ep_100cls` |

## 01. Old full fine-tuning baseline

- Current folder: `01_old_full_finetune_random_3gpu_50ep_head_mismatch`
- Old folder: `output_full_finetune_3gpu`
- Type: historical full fine-tuning baseline
- Model: `spikformer12_512`
- Epochs: 50
- GPU: 3 GPU
- Pretrained checkpoint: none
- Best observed Acc@1: 83.36%
- Final Acc@1: 82.90%
- Note: Later checkpoint inspection showed `head.weight = (1000, 512)` although args had `nb_classes = 100`. Kept as historical reference.

## 02. Baseline MAE pretraining

- Current folder: `02_baseline_pretrain_mae_2gpu_400ep`
- Old folder: `output_pretrain_full_2gpu`
- Type: baseline MAE pretraining
- Model: `spikmae_12_512`
- Epochs: 400
- GPU: 2 GPU
- Important checkpoint: `checkpoint-395.pth`
- Best loss: 0.5812 at epoch 395
- Final loss: 0.5813 at epoch 399

## 03. Baseline checkpoint-395 fine-tuning

- Current folder: `03_baseline_finetune_ckpt395_2gpu_100ep`
- Old folder: `output_baseline_finetune_ckpt395_100ep_2gpu`
- Type: full fine-tuning from baseline pretrained checkpoint-395
- Model: `spikformer12_512`
- Epochs: 100
- GPU: 2 GPU
- Pretrained checkpoint: `02_baseline_pretrain_mae_2gpu_400ep/checkpoint-395.pth`
- Correct 100-class head: yes
- Best Acc@1: 79.62%
- Final Acc@1: 79.62%
- Final Acc@5: 94.20%
- Final test loss: 1.0046

## 04. Updated STDV3 pretraining

- Current folder: `04_updated_stdv3_pretrain_deepmim_logpe_400ep`
- Old folder: `output_updated_stdv3_pretrain_400ep`
- Type: updated STDV3 / DeepMIM + Log-PE pretraining
- Model: `spikmae_deepmim_12_512`
- Epochs: 400
- Final epoch: 399
- Final train loss: 0.8426
- Final reconstruction loss: 0.4330
- Important checkpoints: `checkpoint-395.pth`, `checkpoint-399.pth`

## 05. Updated STDV3 fine-tuning

- Current folder: `05_updated_stdv3_finetune_ckpt75_3gpu_50ep`
- Old folder: `output_updated_stdv3_finetune_ep75`
- Type: updated STDV3 fine-tuning
- Model: `spikformer12_512`
- Epochs: 50
- GPU: 3 GPU
- Best Acc@1: 84.54% at epoch 48
- Best Acc@5: 96.87%
- Final Acc@1: 84.48%
- Final Acc@5: 96.75%
- Final test loss: 0.6134

## 06. Clean 100-class full fine-tuning baseline rerun

- Current folder: `06_clean_baseline_full_finetune_random_3gpu_50ep_100cls`
- Old folder: `output_full_finetune_3gpu_clean_100cls`
- Type: clean rerun of old full fine-tuning baseline
- Model: `spikformer12_512`
- Epochs: 50
- GPU: 3 GPU
- Pretrained checkpoint: none
- Correct 100-class head: yes
- Best Acc@1: 84.16% at epoch 46
- Best Acc@5: 96.42%
- Final Acc@1: 83.95%
- Final Acc@5: 96.70%
- Final test loss: 0.6420

## Archived / unused runs

- Folder: `archived_unused_runs/output_full_finetune_pretrained_v2_partial_epoch27`
- Old folder: `output_full_finetune_pretrained_v2`
- Reason: partial fine-tuning run, stopped around epoch 27, not used as a main result.

## 07. Updated STDV3 v2 pretraining

- Current folder: `07_updated_stdv3_v2_pretrain_binaryqk_xnor_gray2d_logpe2d_2gpu_400ep`
- Type: updated STDV3 v2 pretraining
- Model file: `Updated_STDV3_2.py`
- Main script import: `import Updated_STDV3_2 as Updated_STDV3`
- Model: `spikmae_deepmim_12_512`
- Main idea: Binary Q,K + XNOR + Gray2D + Log-PE 2D
- Epochs planned: 400
- GPU: 2 GPU
- Status: prepared, not started yet

## 08. Updated STDV3 v1 fine-tuning from checkpoint-390

- Current folder: `08_updated_stdv3_v1_finetune_ckpt390_2gpu_100ep`
- Type: updated STDV3 v1 full fine-tuning
- Model: `spikformer12_512`
- Pretrained checkpoint: `04_updated_stdv3_pretrain_deepmim_logpe_400ep/checkpoint-390.pth`
- Pretraining source: Updated STDV3 v1 / DeepMIM + Log-PE
- Epochs planned: 100
- GPU: 2 GPU
- CUDA devices used: `3,7`
- Batch size: 32
- Accum iter: 4
- Fine-tuning recipe: same as baseline checkpoint-395 fine-tuning
- Classes: 100
- Status: running
- Screen session: `updated_stdv3_v1_finetune_ckpt390`
- Script: `run_finetune_updated_stdv3_v1_ckpt390_2gpu_100ep.sh`

## 09. Updated STDV3 v2 debug pretraining

- Current folder: `09_updated_stdv3_v2_debug_pretrain_2gpu_400ep`
- Type: updated STDV3 v2 debug pretraining
- Model file: `Updated_STDV3_2_debug.py`
- Main script import: `import Updated_STDV3_2_debug as Updated_STDV3`
- Model: `spikmae_deepmim_12_512`
- Main idea: revised V2 with attention debug output
- Epochs planned: 400
- GPU: 2 GPU
- CUDA devices planned: `1,5`
- Batch size: 32
- Accum iter: 8
- BLR: 0.00015
- LR: 0.0003
- Weight decay: 0.05
- Warmup epochs: 20
- Mask ratio: 0.5
- Input size: 224
- Dataset: `/media/homes/rafae/uni/datasets/ImageNet_Sub`
- Output folder: `09_updated_stdv3_v2_debug_pretrain_2gpu_400ep`
- Terminal debug output: `terminal_output.txt`
- JSON training log: `log.txt`
- Status: running
- Screen session: `updated_stdv3_v2_debug_pretrain_400ep`
- Started: 2026-05-16 01:32
- Script: `run_pretrain_updated_stdv3_v2_debug_2gpu_400ep.sh`

## 10. Updated STDV3 v1 linear probing from checkpoint-390

- Current folder: `10_updated_stdv3_v1_linear_probe_ckpt390_2gpu_100ep`
- Type: linear probing
- Main script: `main_finetune_linear_probe.py`
- Run script: `run_linear_probe_updated_stdv3_v1_ckpt390_2gpu_100ep.sh`
- Model: `spikformer12_512`
- Pretrained checkpoint: `04_updated_stdv3_pretrain_deepmim_logpe_400ep/checkpoint-390.pth`
- Backbone: frozen
- Trainable part: classification head only
- Epochs planned: 100
- GPU: 2 GPU
- CUDA devices planned: `2,3`
- Batch size: 32
- Accum iter: 4
- Dataset: `/media/homes/rafae/uni/datasets/ImageNet_Sub`
- Output folder: `10_updated_stdv3_v1_linear_probe_ckpt390_2gpu_100ep`
- Terminal output: `terminal_output.txt`
- JSON training log: `log.txt`
- Status: prepared, not started yet
