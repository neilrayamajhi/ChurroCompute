# Churro Findings

Running log of what the metrics actually say about real RL environments.
Newest entries at the top. Every number carries the model, group size, sample count, and date.

Format for each entry:
- **What we ran** — command, env, model, sample size
- **What we measured** — the numbers
- **What it means** — plain-English reading
- **Caveats** — what the number does *not* say

---

## 2026-09-25 — regex-craft difficulty curve on a rented GPU (grader looks broken)

**What we ran**
The full `qwen3-v1` ladder on `liuliu/regex-craft` (0.1.0), on a rented RunPod A40 (48GB, $0.49/hr) instead of the M4:
```
uv run python tools/runpod_ladder.py liuliu/regex-craft
# per model: vf-eval regex-craft -n 30 -r 4 --max-tokens 16384 --timeout 600
# Ollama: OLLAMA_NUM_PARALLEL=4, OLLAMA_CONTEXT_LENGTH=20480
uv run python tools/report_difficulty.py data/raw/regex-craft.jsonl
```
regex-craft only has 18 tasks, so `-n 30` yields 18. Wall-clock: setup 2 min, 0.6b 9 min, 1.7b 16 min, 4b 45 min (hit the per-model limit), 8b 42 min, 14b cut off by the pod's 2.5h self-destruct. Cost ≈ $1.26 for the run.

**What we measured**
| Model | n_tasks | rollouts | pass_rate | signal_rate | dead_too_easy | dead_too_hard |
|---|---|---|---|---|---|---|
| qwen3:0.6b | 18 | 72 | 0.00 | 0.17 ± 0.17 | 0 | 15 |
| qwen3:1.7b | 18 | 72 | 0.00 | 0.11 ± 0.15 | 0 | 16 |
| qwen3:4b | 9 | 36 (+4 timed out, skipped) | 0.00 | 0.00 | 0 | 9 |
| qwen3:8b | 18 | 72 | 0.00 | 0.06 ± 0.11 | 0 | 17 |
| qwen3:14b | 10 | 40 | 0.00 | 0.10 ± 0.19 | 0 | 9 |

Reward components across **every** rollout of **every** model:
| Component | min | mean | max |
|---|---|---|---|
| `regex_correctness` | 0.1 | ≈ 0.49 | 0.5 |
| `match_accuracy` | 0.0 | 0.0 | 0.0 |
| `explanation_quality` | 0.0 | 0.0 | 0.0 |

**What it means — the grader, not the models, sets the score.**

The difficulty curve is flat: every rung averages 0.47–0.50 reward and never exceeds 0.5, so nothing ever "passes". A ~23× jump in parameters (0.6b → 14b) with zero change in score is not a difficulty signal. Two of the three reward components are 0 for all 292 rollouts, so the reward is effectively `regex_correctness` capped at 0.5.

The answers are not wrong. Example: 14b answered `^[a-z]+[0-9]{3}$` for a task whose reference is `[a-z]+\d{3}` — equivalent regexes — and still received `match_accuracy = 0`. This is the Grader Narrowness failure mode (PRD Part 12) showing up before we built that metric: `dead_too_hard` here means "the grader withholds credit", not "the model can't do it".

This probably also explains the earlier regex-craft entry (2026-08-30, 1.7b, 38% signal, 10/16 dead_too_hard): its `mean_spread = 0.15` from rewards clustering around 0.5 fits the same capped reward. That number should not be read as difficulty.

**What the metrics missed**
- `possibly_broken` stayed `False`. It requires `pass_rate == 0` **and** `signal_rate == 0` on every rung, but the 0.1-vs-0.5 wobble in `regex_correctness` gives a few "live" groups, so signal is small but non-zero. A flat-at-a-ceiling reward across the whole ladder (identical mean reward on all rungs, max < pass threshold) should probably also trip it.
- Signal Rate alone would have reported "6–17% signal, mostly too hard" — plausible-sounding and wrong. The ladder view plus a per-component look is what exposed it.

**Caveats**
- 4b and 14b are partial (9 and 10 tasks). 4b writes very long answers (mean ≈ 4.2k output tokens, max ≈ 10k) and hit the 45-min per-model limit; 14b was stopped by the 2.5h budget. A full ladder needs ~4h on this GPU.
- Settings differ from the M4 runs: `--max-tokens 16384` and a 20k context were added because qwen3:4b otherwise generated until the 10-min per-rollout timeout on every attempt. Group size is 4 throughout.
- Only one sample answer was hand-checked for equivalence. Before filing an issue on the environment, check a handful more and read the grader's source for how `match_accuracy` is computed.
- Different hardware (A40 vs M4) and Ollama build; same `qwen3` tags and quantization.

**Ops notes (RunPod)**
- Community-cloud pods with `--public-ip` were never available across 5 attempts; secure-cloud A40/A6000/3090 rented fine.
- Pod setup takes ~2 min; all five models pull in about a minute.
- The pod image's `uv` is too old for the current `prime` CLI, and `prime`'s own dependencies clash with `verifiers 0.3.0` on Linux. Installing the env straight from its Hub index (`uv pip install <env> --extra-index-url https://hub.primeintellect.ai/<owner>/<env>/install/simple/`) avoids both.

---

## 2026-09-17 — gsm8k difficulty curve (Phase 4, first environment)

**What we ran**
Normalized the previously-uncollected 14b rollouts into `data/raw/gsm8k.jsonl`:
```
uv run python tools/collect_run.py "outputs/evals/gsm8k--qwen3:14b/a8d60b1e"
# wrote 444 rollouts (+4 new, 0 deduped)
```
Then ran the difficulty report across the frozen `qwen3-v1` ladder:
```
uv run python tools/report_difficulty.py data/raw/gsm8k.jsonl
```

**What we measured**
| Model | n_tasks | group_size | pass_rate | signal_rate | dead_too_easy | dead_too_hard |
|---|---|---|---|---|---|---|
| qwen3:0.6b | 30 | 4 | 0.60 | **0.43 ± 0.18** | 10 | 7 |
| qwen3:1.7b | 30 | 8 | 0.90 | 0.23 ± 0.15 | 22 | 1 |
| qwen3:4b | 10 | 4 | 1.00 | 0.00 | 10 | 0 |
| qwen3:8b | 10 | 4 | 1.00 | 0.00 | 10 | 0 |
| qwen3:14b | 1 | 4 | 1.00 | 0.00 | 1 | 0 |

Summary flags: `floor=qwen3:0.6b`, `ceiling=qwen3:1.7b`, `best_signal_model=qwen3:0.6b`, `slope=0.30 pass_rate/rung`, `saturated=True`, `possibly_broken=False`.

**What it means — narrow, low-lying difficulty band.**

gsm8k has a real difficulty range, but it lives entirely below the 4B-parameter mark. The interesting story is a two-rung slope from 0.6b (60% pass, 43% signal) to 1.7b (90% pass, 23% signal), then a cliff into total saturation. Every rung from 4b upward returns zero learning signal — training a modern-sized model on gsm8k would waste 100% of compute on dead groups.

The `saturated=True` flag fires correctly at the top rung (14b: 1/1 tasks dead_too_easy). The `best_signal_model` verdict is unambiguous: if you were going to train *anything* on gsm8k with the qwen3 family, it would have to be 0.6b. This matches the intuition that gsm8k is grade-school math and the Qwen3 family solves grade-school math easily above the sub-billion mark.

**What the difficulty metric adds over Signal Rate alone.**
- Signal Rate on 1.7b alone said "23% — meh." The ladder view reveals *why*: gsm8k lives at the wrong difficulty tier for anyone above 0.6b. Signal Rate names the pathology; Difficulty Curve names the fix (use a smaller model, or retire gsm8k as a training env for this family).
- `slope=0.30 pass_rate/rung` is a concrete measure of how sharply capability changes across the ladder. Steeper slopes = the environment's difficulty is highly model-size-sensitive; flatter slopes = something else limits performance (perhaps the grader).
- `saturated` flipping to True only after normalizing the 14b rollouts caught a real bug in interpretation: without top-rung data, the saturation check silently returns False. Missing data ≠ not saturated.

**Caveats**
- **Uneven sample sizes across rungs.** 0.6b and 1.7b have n=30 tasks; 4b/8b have n=10; 14b has n=1. The `pass=1.00, signal=0.00` verdicts at 4b/8b/14b are directionally correct but weakly-powered. A single lucky task ≠ proof of saturation. All three top rungs telling the same story helps, but they're not independent evidence — they're the same claim measured three ways at low n.
- **Mixed group sizes** (0.6b: 4, 1.7b: 8, others: 4). Signal Rate is a rate over groups so it's cross-comparable, but `mean_spread` isn't cleanly comparable across group sizes.
- **The 14b rung wedged early.** vf-eval requested 5 tasks × 4 rollouts = 20 rollouts; only 4 landed (1 complete group). Consistent with previously-logged Ollama+verifiers-0.3.0 wedge issues (see 2026-08-30 iso8601 entry).
- **Ladder is `qwen3-v1` only.** Says nothing about how Llama, Mistral, or Gemma of comparable sizes would score. Cross-family generalization is a B8 concern, not a Phase 4 one.

**Phase 4 progress: 1 of ≥3 environments done.** ROADMAP wants difficulty curves on ≥3 envs before Phase 4 exits. `iso8601-recurrence` and `regex-craft` have only 1.7b data on disk — they need 4 more ladder rungs each before their difficulty reports mean anything. Estimated cost: ~week of wall-clock time per env at ~15min/rollout × ~600 rollouts on the M4. Deferred to a keyboard-attended session.

---

## 2026-09-02 — Phase 3 wrap: three envs, three pathologies, one dead env

**The cross-environment table.** Same model (qwen3:1.7b) throughout.

| Env | n_tasks | group_size | signal_rate | dead_too_easy | dead_too_hard | Pathology |
|---|---|---|---|---|---|---|
| `gsm8k` | 30 | 8 | 23% ± 15% | 22 | 1 | **Saturated** — model too strong |
| `regex-craft` | 16 (partial) | 8 | 38% ± 24% | 0 | 10 | **Partial capability gap** |
| `iso8601-recurrence` | 30 | 4 | **0%** | 0 | 30 | **Complete capability wall** |
| `exact-determinant` | — | — | — | — | — | **Env unrunnable on this stack** (see below) |

**Phase 3 exit criterion met.** ROADMAP asked for signal-rate numbers on ≥3 environments with model + group size stamped. We have 3, with distinct discriminating patterns. The metric works: it doesn't emit the same number for everything, and the number correlates with an observable failure mode (too_easy vs too_hard) that a human reader can act on.

**What the metric does well:**
- Names *why* an env is bad in one glance. Zero live groups tells you "don't train here" without you needing to eyeball rollouts.
- Distinguishes "too strong" from "too weak" via the dead_too_easy/dead_too_hard split. Both are 0 signal, but the fix is opposite (retire tasks vs. use a stronger model).
- Cheap enough to run: 30×4 finishes in ~1h on an M4 for a small env.

**What the metric doesn't do:**
- Can't tell you if the grader itself is broken (see: iso8601 at 100% dead_too_hard — is the model bad or is the grader rejecting valid answers? PRD Part 12 "Grader Narrowness" will answer this).
- Wald CI collapses to 0 at p=0, understating uncertainty for capability-wall envs.
- One-model snapshot. The whole picture requires the ladder (Phase 4).

---

## 2026-09-02 — exact-determinant × qwen3:1.7b (env unrunnable on this stack — logged as finding)

**What we tried**
```
vf-eval exact-determinant -m qwen3:1.7b -n 30 -r 4 -s ...   # 30-min stall, 0 rollouts written, killed
vf-eval exact-determinant -m qwen3:1.7b -n 30 -r 2 -s ...   # ~62-hour stall, 0 rollouts written, killed
```

**What happened**
Both attempts wedged before writing a single rollout. Server-side symptoms: hundreds of worker heartbeat timeouts (551 in the r=2 attempt), "Active tasks" counter grew beyond the requested rollout count as verifiers re-queued failed tasks, Ollama's `expires_at` occasionally advanced (proving *some* inference happened) but never enough to complete a group and flush results.

**Root cause hypothesis.** exact-determinant sends matrix inputs that, combined with CoT determinant reasoning, blow past Ollama's default `-c 4096` context window. Requests either fail server-side or return truncated garbage the grader rejects, worker retries, worker dies, restart loop.

**What it means as a Churro finding.**
- **Environment install-failure/runtime-failure rate is itself the finding**, per ROADMAP Phase 7 and Non-Negotiable #4 ("fail loudly").
- Running total for envs I've attempted with this stack: 4 broken (`apex-shortlist`, `aime2026`, `html-extraction`, `exact-determinant`), 4 working (`gsm8k`, `if-rlvr`, `regex-craft`, `iso8601-recurrence`). **~50% failure rate on a small sample.** Consistent with ROADMAP's prediction that Phase 7 (fingerprint everything) will spend a lot of time logging broken envs, not measuring them.
- The report card for exact-determinant is not `signal_rate=?` — it's `status=unrunnable, reason=context-window-overflow-suspected`. This is a distinct fingerprint category we should design for when we get to Phase 6 (packaging).

**No `data/raw/exact-determinant.jsonl` was written.** Nothing to report; nothing to hide.

---

## 2026-08-30 — iso8601-recurrence × qwen3:1.7b (total capability wall)

**What we ran**
```
vf-eval iso8601-recurrence -m qwen3:1.7b -n 30 -r 4 -s --api-base-url http://localhost:11434/v1
```
- Full completion: 30 tasks × 4 rollouts = 120 rollouts.
- Reduced from r=8 to r=4 to dodge the vf-eval-on-Ollama wedge that hit gsm8k and regex-craft. Finished in ~1h 15min without a single worker restart.

**What we measured**
| Metric | Value |
|---|---|
| `signal_rate` | 0.0 (±0, degenerate Wald CI) |
| `n_tasks` | 30 |
| `group_size` | 4 |
| `dead_too_easy` | 0 |
| `dead_too_hard` | **30** |
| `mean_spread` | 0.0 |

**What it means — total capability wall.**

**Every single rollout scored 0.0.** qwen3:1.7b cannot do a single ISO 8601 recurrence rule task even partially. Every task in the sample became a `dead_too_hard` group; zero live groups means zero learning signal for GRPO training. If you tried to train qwen3:1.7b on this environment, **100% of your compute would be wasted** — the model would never get a non-zero gradient anywhere.

This is the "capability wall" pattern that Signal Rate is designed to name in one number:
- **regex-craft** at 62.5% dead_too_hard was a *partial* wall — the model got some tasks sometimes.
- **iso8601-recurrence** at 100% dead_too_hard is a *complete* wall — the model gets nothing, ever. Different phenomenon, correctly distinguished.

**The CI degeneracy is honest.** `signal_rate_ci95=0.0` is not a bug — the Wald formula `1.96 × sqrt(p(1-p)/n)` collapses to zero when p=0. It's the formula telling us "no observed live groups, so no uncertainty in the estimate that the rate is 0" (which is technically true for this sample but understates the epistemic uncertainty — with n=30 we can't rule out a *true* rate of a few percent). **Worth logging as a limitation of the Wald estimator we chose.** PRD Part 7 accepts this tradeoff for simplicity; a Wilson or Clopper-Pearson interval would give a non-zero upper bound at p=0.

**Caveats**
- `group_size=4` not 8 — deliberately reduced to avoid the vf-eval wedge. Signal rate itself is a rate over groups so it's directly comparable to gsm8k (r=8) and regex-craft (r=8), but `mean_spread` and CI aren't cleanly comparable across different group sizes.
- Could the grader be broken? At `dead_too_hard=30/30` you always have to ask: is this the model failing, or the grader rejecting everything? To disambiguate, would need to eyeball a few rollouts and see if the completions look plausible-but-wrong (model failure) vs. plausible-and-right (grader bug). Not done yet — that's a Phase 5 "Consistency" metric task.

**Speed note.** The `-r 4` reduction — halving rollouts per task — finished cleanly in ~1h 15min. Compare: r=8 runs wedged at ~50% completion in ≥8h. **Preliminary rule for this Mac+Ollama+verifiers stack: r=4 is the practical ceiling for local runs.**

---

## 2026-08-30 — regex-craft × qwen3:1.7b (partial run, first cross-env comparison)

**What we ran**
```
vf-eval regex-craft -m qwen3:1.7b -n 30 -r 8 -s --api-base-url http://localhost:11434/v1
```
- Killed after 8h wall-clock when vf-eval worker wedged at 128/240 rollouts (same failure mode as gsm8k run).
- Clean stop between task boundaries: 16 complete tasks × 8 rollouts each.

**What we measured**
| Metric | Value |
|---|---|
| `signal_rate` | 0.375 (±0.2372, 95% Wald CI) |
| `n_tasks` | 16 (partial; target was 30) |
| `group_size` | 8 |
| `dead_too_easy` | 0 |
| `dead_too_hard` | 10 |
| `mean_spread` | 0.1527 |

**What it means — the metric discriminates.**

| | gsm8k | regex-craft |
|---|---|---|
| signal_rate | 23% | **38%** |
| dead_too_easy | 22/30 | **0/16** |
| dead_too_hard | 1/30 | **10/16** |
| dominant failure mode | too easy | **too hard** |

Same model, same group size, opposite failure modes. gsm8k is a *saturation* problem (model too strong); regex-craft is a *capability* problem (model too weak). Zero dead_too_easy tasks in regex-craft: the model didn't consistently ace a single regex task in the sample. This is exactly what a working discrimination metric should show — different envs have different pathologies, and Churro names them.

**Reward-shape observation.** `mean_spread=0.153` is less than half of gsm8k's `0.355`. That's not because regex-craft's live groups are more marginal — it's because regex-craft has a **graded reward** (0.0 / 0.5 / 1.0 ish), while gsm8k is effectively binary. When live-group scores cluster around a single non-extreme value like `0.5, 0.5, 0.5, 0.4, 0.5, 0.6, 0.5, 0.5`, the stdev drops. **This is a preview of what the Reward Shape metric (PRD Part 10) will measure directly** — reward distributions with a lot of decimal values but few effective bins are a real category worth flagging.

**Caveats**
- **n=16 not n=30.** CI is ±24%, so the true signal_rate could be anywhere in [14%, 62%]. The direction (higher than gsm8k) is real, but the magnitude is soft.
- Partial run because vf-eval wedged at ~50% completion — same class of bug as the gsm8k run. Not a Churro issue but a verifiers-0.3.0-on-Ollama issue.
- Next run (iso8601-recurrence) reduced to `-r 4` to try to avoid the wedge.

---

## 2026-08-29 — gsm8k × qwen3:1.7b (first real Signal Rate)

**What we ran**
```
vf-eval gsm8k -m qwen3:1.7b -n 30 -r 8 -s --api-base-url http://localhost:11434/v1
```
- 30 tasks × 8 rollouts = 240 inferences
- Elapsed: ~22 hours wall-clock on M4 (Mac slept overnight; awake time ~4h)
- Mean generation: 16m 55s per rollout, ~2,017 output tokens per rollout (chain-of-thought)

**What we measured**
| Metric | Value |
|---|---|
| `signal_rate` | 0.2333 (±0.1514, 95% Wald CI) |
| `n_tasks` | 30 |
| `group_size` | 8 |
| `dead_too_easy` | 22 |
| `dead_too_hard` | 1 |
| `mean_spread` | 0.3549 |

**What it means**
- ~77% of any GRPO training compute on this env with this model would be wasted (dead groups produce zero gradient).
- 22/30 tasks: model got them 8/8 correct. These tasks would be retired in real training.
- 1/30 tasks: model missed 8/8. Either genuinely hard for a 1.7B model or the grader is off (would need to eyeball rollouts to tell).
- 7 live groups had a mean within-group stdev of 0.355 — rewards are binary, max stdev = 0.5, so live groups are close to 50/50 splits. Strong signal on the live tasks (not the marginal `[0.51, 0.49, ...]` case the PRD warns about).
- Contrast: the Phase 2 6-rollout sample showed 0% signal. Misleading. With proper GRPO-scale sampling, the real picture (~23%) emerges. **Vindicates the ROADMAP's insistence on n=30 × r=8 and error bars.**

**Caveats**
- Single environment, single model. A metric that reports the same number for one env doesn't prove anything about discrimination. Need ≥2 more envs before claiming the metric works.
- Wide CI (±15%) — true rate is somewhere in [8%, 38%]. n=30 is small; if we bump to n=100 the CI tightens.
- Wald CI degenerates near p=0 or p=1. For signal_rate ≈ 0.23 we're safely in the middle.
- Relative to qwen3:1.7b specifically. Same env with qwen3:0.6b might show ~80% signal; with qwen3:14b maybe 5%. That's what Phase 4 (Difficulty Curve) will measure.

**Operational lessons**
- `verifiers` 0.3.0 does NOT read `OPENAI_BASE_URL`. Must pass `--api-base-url http://localhost:11434/v1 --api-key-var OPENAI_API_KEY` explicitly. First two attempts silently hit Prime Intellect's cloud (default `--provider prime`) and 404'd 240 times.
- gsm8k CoT rollouts average ~2K output tokens → ~15 min/rollout on M4 with qwen3:1.7b Q4_K_M. Budget accordingly: a single 30×8 run ≈ 2–4 hours awake wall-clock.
- Mac sleep kills worker heartbeats. verifiers restarts the worker automatically but wastes time. Disable sleep before long runs (or wire in `caffeinate`).

---

## 2026-08-28 — gsm8k × qwen3:1.7b (Phase 2 sample, n=3 × r=2)

**What we ran**
```
vf-eval gsm8k -m qwen3:1.7b -n 3 -r 2 -s
```
Tiny 6-rollout smoke test to prove the pipe worked end-to-end.

**What we measured**
- `signal_rate: 0.0000` (0/3 groups live, all 3 dead_too_easy)
- All 6 rollouts scored 1.0

**What it means**
- Interpreted at the time as "gsm8k is fully saturated for qwen3:1.7b" — turned out to be a sampling artifact (n too small, tasks happened to be easy ones).
- The real number from a proper 30×8 sample is 23%, not 0%. **Lesson: never publish a signal rate from n<30 without loud caveats.**

**Caveats**
- n=3 tasks is far below the ROADMAP-mandated n=30. This was a smoke test, not a measurement. Left here as an object lesson in why sample size matters.

---

# Cross-environment comparisons

_Empty. Fill in as we run signal_rate on additional envs. The whole point of the metric is discrimination — the interesting story is when env A and env B differ by 20+ percentage points on the same model, or when the same env differs by 20+ pp across a model ladder._

Planned:
- [ ] `wordle` × qwen3:1.7b (PRD Part 7 uses this as its example — good sanity check)
- [ ] one more Prime Intellect Hub env of a different flavor (coding? logic?)

# Cross-model comparisons (Phase 4 preview)

_Empty. Once the model ladder is pulled, run gsm8k across qwen3:{0.6b, 1.7b, 4b, 8b, 14b} and record the curve here. Signal rate should peak at some middle model and collapse at both ends (too weak = all-fail, too strong = all-pass)._

---

# Meta-findings about Churro itself

- **The 6-row → 240-row gap.** The Phase 2 smoke test reported 0% signal; the Phase 3 real sample reported 23%. A 23-percentage-point swing between "smoke test" and "measurement" is exactly the failure mode Churro exists to expose in *other people's* eval reports. Score bars matter.
- **CoT-heavy envs are expensive to fingerprint.** 2 hours per env × 200 envs (Phase 7's target) = 400+ machine-days on this M4. Fingerprinting the whole Hub will need either concurrent rollouts, a beefier machine, or accepting smaller n.
