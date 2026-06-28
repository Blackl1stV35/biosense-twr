# BioSense TWR — Continuous Training Objective

## Goal

Continuously improve `poc/notebooks/A100_sim_run.ipynb` until the model
reaches and sustains the following targets across a full 2,000-scenario run:

| Metric | Target |
|---|---|
| Best OKS | ≥ 0.98 |
| Mean OKS (last 50 batches) | ≥ 0.92 |
| Loss spikes above 1,000 | 0 occurrences |
| Session resumability | Full restore of weights + OKS baseline from Drive |

Current best achieved: OKS 0.9902 (Run 2), regressed to 0.10–0.25 (Run 3)
due to missing weight restore on session reconnect. That fix is already
specified — the remaining work is stabilisation and hardening.

---

## Active issues to resolve, in priority order

### P0 — Gradient explosion (loss spikes to 14,000)

Loss spikes every ~20 batches indicate unbounded gradients on hard outlier
scenarios. Add gradient clipping immediately after
`scaler.scale(loss).backward()` in Cell 6:

```python
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Do not change any other part of the training loop.

### P1 — Session weight restore (regression from Run 2 → Run 3)

Cell 5: after `model = BioSensePoseNet(...).to(DEVICE)`, insert:

```python
_best_model_path = Path(DRIVE_ROOT) / 'model_best.pt'
if _best_model_path.exists():
    model.load_state_dict(torch.load(str(_best_model_path), map_location=DEVICE))
    print(f'Restored weights from {_best_model_path}')
else:
    print('No saved model found — training from scratch.')
```

Cell 6: at top of loop setup, insert:

```python
_oks_log_path = Path(DRIVE_ROOT) / 'best_oks.txt'
best_oks_so_far = float(_oks_log_path.read_text()) if _oks_log_path.exists() else 0.0
```

In the best-model save block, persist OKS alongside weights:

```python
if oks > best_oks_so_far:
    best_oks_so_far = oks
    torch.save(model.state_dict(), Path(DRIVE_ROOT) / 'model_best.pt')
    Path(DRIVE_ROOT, 'best_oks.txt').write_text(str(best_oks_so_far))
    print(f'[Best model saved] OKS: {best_oks_so_far:.4f}')
```

### P2 — Alpha scheduling (loss weight progression)

Once mean OKS exceeds 0.70 for 10 consecutive batches, increase alpha:

```python
if len(oks_scores) >= 10 and np.mean(oks_scores[-10:]) > 0.70:
    criterion.alpha = 5.0
if len(oks_scores) >= 10 and np.mean(oks_scores[-10:]) > 0.85:
    criterion.alpha = 10.0
```

Add this check immediately after `oks_scores.append(oks)` in Cell 6.

### P3 — Two-wall stress test

After mean OKS (last 50) exceeds 0.92 on mixed scenarios, add a dedicated
evaluation pass over two-wall reinforced concrete scenarios only:

```python
# End of Cell 6, after main loop
stress_scenarios = [
    {'mat': 'reinforced_concrete', 'n_walls': 2, 'thickness': 0.30,
     'range_m': r, 'activity': a}
    for r in [5.0, 10.0, 15.0, 20.0, 25.0]
    for a in ['walk', 'crouch', 'stand']
]
```

Run inference only (no training) on these 15 scenarios and log OKS per
range. This closes the gap to the BENCHMARK.md comparison table.

---

## Rules for the agent

1. **Only edit** `poc/notebooks/A100_sim_run.ipynb` and
   `poc/biosense_twr/inversion/pose_net.py`. Do not touch any other file.

2. **Apply P0 and P1 first** before any other change. Verify by reading
   the relevant cells back after editing.

3. **Do not change** the pipeline order, module imports, Rust kernel calls,
   checkpoint chunk size, or Drive folder structure.

4. **After each edit**, state which cell was changed, what line was changed,
   and what the expected effect on OKS or loss is.

5. **Stop and report** if any of the following occur:
   - A cell cannot be edited without restructuring another cell
   - Mean OKS (last 50) exceeds 0.92 AND loss spikes are eliminated
   - Drive budget drops below 0.5 GB remaining

6. ~~**Do not re-run the notebook.**~~ **Authorised to run the notebook** when
   the researcher explicitly requests execution or evaluation.

---

## Reference: what is working and must not change

- Coordinate normalisation (pelvis-centred, head-pelvis scale)
- Range-gated voxel grid per scenario (`y_range = range_m ± 3.0`)
- OKS sigma = 0.5
- PoseLoss alpha starts at 1.0 (scheduled up by P2)
- Chunked checkpoint → Drive every 200 scenarios
- `model_best.pt` saved on every OKS improvement

---

## Completion criteria

The agent's work is complete when a single run produces:

- Best OKS ≥ 0.98  
- Mean OKS last 50 ≥ 0.92  
- Zero loss spikes above 1,000  
- `model_best.pt` on Drive reflects these weights  
- Two-wall stress test OKS logged per range band