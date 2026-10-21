# Project notes

Keep notes here, not in Claude's memory directory: this VM wipes everything outside `~/wash` (this folder, `/lambda/nfs/wash`) when it shuts down.

## User
- Timezone is America/Vancouver; the VM clock is UTC. Only send reminder push notifications between 12pm and 12am local time (19:00–07:00 UTC during PDT, 20:00–08:00 UTC during PST from Nov 1).

## Research constraint
- The research question is about diversity from a FIXED input (system prompt + product name only). Never add style directions or any other variation to the input to get diversity; it has to come from the policy itself.

## Sampling: training vs eval
- Training rollouts and the fixed-prompt eval both sample with temperature 1.0, top_k=50 (`top_k=50` in `GRPOConfig`; HF `generate` default for eval). top_k=50 is the deployment setting; use it too when benchmarking the trained model (e.g. against GPT).
- History: rollouts used top_k=0 (full vocabulary) until 2026-10-21. With the 2B model that made rollouts much noisier than eval, and the run degraded into instruction-style prompts ("Create a wild poster for X, do not ..."), so training was switched to top_k=50.

## Running
- Training runs in the `grpo-vllm` conda env (a clone of base with vLLM 0.32, which pins torch 2.13 / transformers 5.19, plus flash-linear-attention). vLLM generation is currently off: quality over speed. The HPSv3 reward server runs from `~/.hpsv3`.
- Policy is Qwen3.5-2B (the 4B followed the system prompt better but collapsed onto one default style), full fine-tuning with torchao 8-bit AdamW and gradient checkpointing. No KL penalty (`beta=0`).
- `./launch.sh` starts or reuses the HPSv3 server, then launches `main.py` into the next `logs/train_N.log`, resuming from the latest checkpoint in `trainer_output/`.
- Delete `trainer_output/wandb_run_id.txt` (and move old checkpoints aside) to start a fresh run instead of resuming.
