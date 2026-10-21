# Night of 2026-10-21: what happened to the training runs

All runs: Qwen3.5-4B, full fine-tuning (torchao 8-bit AdamW), group size 8, top_k=50 rollouts, thinking off, HF generation.
Old checkpoints are in `old_runs/runNN/`.

| run | change | what went wrong | when |
|---|---|---|---|
| 13 | text style diversity as a running-normalized probability | drifted to clean retail ads (prices, "BUY NOW", "50% OFF"); image anti-design 0.79 → 0.53 | ~step 30 |
| 14 | both diversity terms scaled by anti-design | anti-design went up (0.74 → 0.85–0.89), but prompts ran away to 3,000–10,000 characters | ~step 12 |
| 15 | prompts over 3000 chars earn 0 from every judge; max 1024 tokens | drifted to minimalist Swiss-style posters described as "anti-design"; `%%` dropped (14/16 → 2/16), negations and "should embrace…" phrasing came back; text reward ~1.0 → 0.15 while image anti-design stayed ~0.67 | ~step 24 |
| 16 | learning rate 1e-5 → 3e-6 | running now | |

## The common pattern

At lr 1e-5 the policy moves within 15–35 steps to wherever the judges currently overlap most. Each fix closed one loophole and the next one appeared. Run 16 lowers the step size instead of adding another reward patch.

## Open reward-side options (not applied; for you to decide)

1. **Text and image judges disagree on minimalist posters.** In run 15 the text judge scored the Swiss-minimal prompts low (anti-design 0.42/3), but the image judge still gave the resulting posters ~0.67, and the image side carries more weight. Options:
   - tighten the image `anti_design` question so clean grid and Swiss layouts count as conventional, the same way the text judge already does;
   - or make the image reward depend on the text gate too (e.g. multiply by the text `describes_image`/`no_negation` gate), so a prompt that the text judge rejects cannot be rescued by the image side.
2. **Keep the negative prompt alive.** `%%` is only worth 0.5 in the format reward and was dropped as soon as something else paid more. Options: raise it back to 1.0, or gate the text reward on having exactly one `%%`.
3. **KL penalty.** A reference model does not fit next to Z-Image + HPSv3 with full fine-tuning. LoRA would make it nearly free (the reference model is the base without the adapter), at the cost of switching away from full fine-tuning.
4. **Style collapse in general.** Every run settled on one look per phase (neon slime → grunge sale collage → retail → Memphis geometric → Swiss minimal). The diversity terms slow this down but don't stop it. A bigger lever would be conditioning: e.g. give each prompt a randomly sampled style direction or design movement in the user message, so diversity comes from the input rather than only from the reward.

## Monitoring tools (scratchpad, lost when the VM shuts down)

- `check.py`: degradation scan (ignores quoted ad copy), wandb diversity curves, anti-design/validity tracker on the newest 16 images.
- An earlier version of the tracker paired images with the wrong step's product (Z-Image skips over-length prompts, and images of the step in progress exist before its completions file). Fixed at 13:22 UTC; the two affected log entries are marked.

## Run 16 status (lr 3e-6), as of 15:05 UTC, step 48

- Healthy on every guard: anti-design 0.72–0.84 (image judge, newest 16), validity ~0.75–0.9 (dips only on abstract products like "funeral service", "air", and one "credit card" step), `%%` in 22–24/24, no instruction-style prompts, length ~200–250 tokens, entropy ~1.6–1.75.
- Products and quoted copy fit well ("TOFU IS THE NEW GOLD", "BARK BEFORE YOU DINE", "FORGET YOUR HARD DRIVE").
- But style has converged onto one look for every product: acid green + magenta, black spray paint, sticker-collage chaos, dripping graffiti type. This is option 4 above (style collapse); the diversity rewards slow it but do not prevent it. Left running since it is not one of the stop conditions.
