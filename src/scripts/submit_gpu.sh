#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --account=jgray21_gpu
#SBATCH --time=72:00:00
#SBATCH --qos=qos_gpu

#### execute code
export HYDRA_FULL_ERROR=1

python ../src/run.py model=score_model datamodule=docking_datamodule logger.wandb.project=dock_diffusion trainer.max_epochs=30 trainer.check_val_every_n_epoch=1 callbacks.model_checkpoint.save_top_k=5 trainer.gradient_clip_val=0.0 datamodule.train_dataset=pinder_train datamodule.val_dataset=pinder_val
