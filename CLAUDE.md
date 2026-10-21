# Project notes

Keep notes here, not in Claude's memory directory: this VM wipes everything outside `~/wash` (this folder, `/lambda/nfs/wash`) when it shuts down.

## User
- Timezone is America/Vancouver; the VM clock is UTC. Only send reminder push notifications between 12pm and 12am local time (19:00–07:00 UTC during PDT, 20:00–08:00 UTC during PST from Nov 1).

## Sampling: training vs eval
- Training rollouts (TRL → vLLM) sample with temperature 1.0 and top_k=0 (full vocabulary) on purpose, for more exploration during training.
- The fixed-prompt eval uses HF `generate` defaults (top_k=50). This is the deployment setting, so eval images represent production quality and are expected to look cleaner than rollouts.
- Don't add top_k to `GRPOConfig` to "fix" rollout quality. When deploying or benchmarking the trained model (e.g. against GPT), sample with top_k=50.

## Running
- Training runs in the `grpo-vllm` conda env (vLLM 0.32 pins torch 2.13 / transformers 5.19, so it is kept out of the base env). The HPSv3 reward server runs from `~/.hpsv3`.
- `./launch.sh` starts or reuses the HPSv3 server, then launches `main.py` into the next `logs/train_N.log`, resuming from the latest checkpoint in `trainer_output/`.
- Delete `trainer_output/wandb_run_id.txt` (and move old checkpoints aside) to start a fresh run instead of resuming.
