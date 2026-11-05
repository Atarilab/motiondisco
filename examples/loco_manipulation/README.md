# Loco-Manipulation Contact Planning Example

Shinka task: generate a `contact_mode_seq` that jointly encodes the locomotion gait and manipulation contacts for a humanoid robot performing a long-horizon pick-and-place task.

## Task

The planner produces a `contact_mode_seq` dict whose channels describe
how the robot's end-effectors, soles, and manipulated objects make and
break contact across a sequence of modes:

```python
# Schema (per channel):
#   EEs / soles:  (env_id, contact_type)            or () for free
#   Objects:      (self_id, env_id, contact_type)   or () for free
#   contact_type: 1 = unilateral (only value supported for now)

contact_mode_seq = {
    "left_ee":    [(),     (9, 1),  ()    ],   # grasp box1 feat 2 (global env_id 9)
    "right_ee":   [(),     (),      (11,1)],   # grasp box1 feat 4 (global env_id 11)
    "left_sole":  [(1, 1), (1, 1),  (1, 1)],   # stand on floor
    "right_sole": [(1, 1), (1, 1),  (1, 1)],
    "box1":       [(6,1,1),(),      ()    ],   # box1 self-feat 6 resting on floor env_id 1
}
```

There are two distinct ID spaces in every contact tuple:

| Space | Meaning |
|---|---|
| `env_id` | global feature index (used in every tuple as the env_id) |
| `self_id` | per-object feature index (used only in object channels) |

The exact ranges are scene-dependent and are introspected at run
time from the active task's compiled faro scene. They are
also pasted into the LLM system prompt automatically — see the
"Switching tasks" section below for how to swap scenes without
touching code.

## Switching tasks

A task is three strings — two registry names resolved against
faro (`scene`, `initial_condition`) plus a free-form LLM
`description`. Scene IDs are dotted module paths into faro's
scene families (`motiondisco.long_dist_pick_place`, `faro.box_pushing`);
MotionDisco's own scenes live under `examples/loco_manipulation/scenes/`
(a namespace package merged with faro's) and keep bare names
(`truck_unload_stack`, `floor_stack_carry`, `climb_table_up_down_with_box`
— a copy of faro's scene whose discrete goal drops the box entry
and has no target, with the box required on the floor 0.25–0.85 m past the
table's far edge). Each scene's `goals.py`
exposes `DISCRETE_GOALS` (final-mode gate), `TERMINAL_GOALS` (KSO / TO
target with per-solver `kso=` / `to=` modes) and `OUTCOME_SCORERS`; the
first non-empty entry of each is used. The task block lives in the
config:

```yaml
task:
  scene: example_scene
  initial_condition: default
  description: "pick box1 from the floor and place it on table1 ..."
```

Scene + IC are looked up via `scenes.common.registry.{make_scene,
get_initial_condition}` against
`submodules/faro/src/scenes/<scene>/{scene,initial_conditions}.py`;
goals are discovered from `scenes/<scene>/goals.py`. Scenes with no
`goals.py` (or only empty/`none` factories) resolve to "no discrete
filter, zero continuous cost". `description` is prompt prose only.

**Hydra launcher:** edit `shinka/configs/task/loco_manipulation.yaml`,
or override at the CLI:

```bash
motiondisco_launch task=loco_manipulation \
    task.initial_condition=<other_ic> \
    task.description="'short task prose'"
```

Multirun sweep (tries each variant in turn):

```bash
motiondisco_launch -m task=loco_manipulation \
    task.scene=example_scene,motiondisco.banana_real
```

**Standalone runner:** edit the `task:` block at the top of
`shinka_small.yaml` / `shinka_ollama.yaml`, then
`python run_evo.py --config_path shinka_small.yaml`.

**Adding a new task**: register the IC upstream in
`submodules/faro/src/scenes/<scene>/initial_conditions.py`
(listed in its `INITIAL_CONDITIONS` dict) and declare goals in
`scenes/<scene>/goals.py` via its `DISCRETE_GOALS` / `TERMINAL_GOALS` /
`OUTCOME_SCORERS` dicts; `_task_loader` uses the first non-empty entry
of each. No new files on the shinka side.

The LLM system prompt is auto-built from the live scene's face index
(`scene.faces`), `env_objects`, and `resolved_state.contact_mode` —
see `prompts.build_task_sys_msg`.

faro keys contacts as `("ee", robot_id,
patch)` / `("obj", name)`. The LLM-facing plan keeps flat channel
names (`left_ee`, `right_sole`, `white_box`, ...); `_task_loader`
converts at the solver boundary (`to_faro_plan`, via faro's
`import_legacy_contacts`). Only single-robot scenes are supported.

## Scoring

Proposals are scored in three sequential stages:

1. **Discrete goal match** — the final mode of `contact_mode_seq`
   must satisfy the task's `discrete_goal`. If not, `combined_score = 0`.
2. **IK feasibility** — faro's KSO (kinematic sequence
   optimization, formerly "seqik") checks that the contact sequence is
   kinematically reachable. If infeasible, `combined_score = 0`. A
   terminal goal with `kso="hard"` is tried hard first, then soft.
3. **TO cost** — trajectory optimization (`to_eval.solve_to`, shared
   with `run_to_all.py` / `recompute_to_costs.py`) computes a dynamic
   cost (constraint violations + control smoothness). The terminal
   goal's `to=` mode picks soft / hard / off. Lower cost = higher score.

| Outcome | `combined_score` |
|---|---|
| Discrete goal not matched | `0.0` |
| IK infeasible / out of range / empty plan | `0.0` |
| Goal + IK pass, TO returns cost `c` | `1 / (1 + c)` |

### Solver feedback

`violation_feedback.py` translates the solvers' constraint violations into
the plan's own vocabulary, and `evaluate.py` adds them to the LLM's text
feedback as a "Solver feedback" block (first line also in
`public_metrics["violations"]`):

- **KSO** (plan failed every full-plan attempt): rows
  `k=<mode>:<role>:<name>[<row>]` from `get_performance_metrics()` are
  mapped to mode, channel (`left_ee`, `big_box`), touched face and meaning
  (gap / pushed in, tilt, slip, drift, overhang, collision with link names,
  goal error). The headline ("KSO infeasible: first break at mode 20 of 34
  (left_ee -> big_box left); modes 0–19 solve on their own.") and the first
  bullets describe the same failing prefix, so they cannot disagree; the
  full-plan solves' violations follow, marked as secondary. Attempts are
  pooled; problems recurring at the same mode in several solves rank first,
  one-off problems are reported as solver noise. An object kept off its
  support while it penetrates another body becomes one diagnosis with the
  clearance geometry ("big_box ends up on top of rear_step … rear_step
  spans x 0.40–0.80 …"); an object that moves before anything touches it
  is reported as a reach problem, not a slip. A goal miss (every contact
  works without the target) reports final position, distance and
  tolerance; the repair probe adds the walk insertion that fixed the plan
  when one did.
- **TO** (always, when it ran): moto rows (`node // nodes_per_mode` = mode)
  for contact kinematics, wrench cone (friction, torsion, tipping, pulling),
  switch forces, object force balance, torque / joint limits, timing and
  collisions, plus the solver status (1 = out of iterations, 2 = numerical
  failure). TO inequality values include the interior-point slack, so they
  over-state small violations. When TO did not converge, `step_feedback.py`
  also measures how far each foot travels when it lands in TO's solution
  and reports stretches (between grasps / set-downs) with steps over 0.8 m
  of foot travel, twice the 0.4 m step length, with one extra walking step
  per long step before the next grasp or set-down ("modes 24–37: in 2 of
  9 steps a foot travels more than 0.8 m … give it about 2 more walking
  steps before mode 38 (big_box set down on floor surface)"). KSO cannot
  catch this: it has no limit on how far a foot moves between modes. Long
  steps alone do not stop TO converging on flat ground, so it is not
  reported for converged plans. `grip_feedback.py` adds, likewise only on
  failure, any carried object held by both hands on faces at right angles
  (front + right, top + back) or on the same face: the hands cannot squeeze
  it, so friction alone holds its weight (stacked objects included) and it
  slips and tilts. No such grip converged in the stored runs (0 of 2,
  against 66 of 81 opposite-face grips on long_dist_pick_place). It is a
  hint, not a rejection, so unusual grips stay open to the search.

Robot-link collision names are SHA-256 digests in faro; they are resolved
through the task's `collision_index`. Raw records land in `extra.json`
(`kso_violations`, `to_violations`, `to_status`). `SHINKA_HIDE_IK_FEEDBACK=1`
hides all of it.

The five structural rules (`well_formed`, `equal_length`,
`at_least_one_sole`, `no_noop_mode`, `object_passive`) and the
`efficiency` term are still computed and reported as private
diagnostics, but they do not contribute to ranking.

### Quality scoring (opt-in: `task.score_mode=quality`)

The table above is the default `legacy` rule used in the paper's
ablations. It ignores TO convergence, ignores how well the task was
achieved (e.g. `motiondisco.box_stacking` rewards any smooth plan regardless
of stack height), and averages constraint violation over the horizon,
so padded plans look better. `score_mode=quality` (`scoring.py`)
replaces it with ordered bands:

| Outcome | `combined_score` |
|---|---|
| TO converged | `0.1 + 0.9 * task_quality * to_quality` |
| IK feasible, TO failed / not converged | `0.1` |
| IK feasible, TO disabled | `0.1 + 0.9 * task_quality` |
| IK goal missed (every contact works without the target) | `0.1 * (0.9 + 0.1 / (1 + miss distance))` |
| IK infeasible | `0.1 * 0.9 * events_feasible / events_total` (contact events before the break; walking steps do not count) |
| goal not matched, < 2 modes, duplicate, over the cap | `0` |

`to_quality = 1 / (1 + 1e2 * max_violation + 1e-2 * jerk / modes)` (the
worst violation, so it is neither diluted by padding nor inflated by long
horizons). `task_quality` maps the scene's outcome scorer
to (0, 1]: `1 / (1 + residual)` for goal residuals, `h / (h + h0)` for
higher-is-better scorers listed in `scoring.SCORER_DIRECTION`
(`box_stack_height`; `h0` = value at the initial state). The LLM prompt
describes whichever rule is active.

## Diverse trajectories

Two levels: the search finds different *strategies* (which contacts,
in which order), and a post-search sweep varies how each strategy is
*executed*.

**Level 1 — search** (opt-in, `variant=loco_diverse`):

```bash
motiondisco_launch task=loco_manipulation variant=loco_diverse cluster=local \
    task.scene=motiondisco.box_stacking task.score_mode=quality
# or, through the search script:
SCENES=motiondisco.box_stacking DIVERSE=1 SCORE_MODE=quality REPEATS=3 \
    ./examples/loco_manipulation/run_search.sh
```

- `task.diversity=true` (set by the variant) makes `evaluate.py` emit
  `plan_patches` (which faces the hands touch and which surfaces feet
  and objects rest on, order ignored — the archive's descriptor key),
  `plan_signature` (the strategy: contacts made and broken in order,
  walks collapsed so step counts do not matter — `plan_signature.py`),
  `plan_novelty` (patch-set novelty weighted above order novelty, with
  the contacts every plan shares — initial state, discrete goal —
  excluded) and a text-feedback list of every explored patch set and
  order.
- `database/island_diverse.yaml`: 4 isolated islands (no migration, no
  dynamic spawning) with `weighted` and `cell_rarity` parent selection
  (`cell_rarity` favours elites of rarely visited patch cells), and a
  MAP-Elites style archive (`archive_selection_strategy: descriptor`)
  that keeps the best plan per `plan_patches`: inside a cell the score
  decides, `plan_novelty` only breaks cross-cell eviction, and
  `archive_min_score: 0.0` keeps failed plans out of the archive and the
  parent pools.
- The variant also sets Opus + Sonnet with mixed reasoning efforts,
  crossover patches, contact-strategy full-rewrite instructions
  (`prompts.full_rewrite_variants`), one neutral exploration axis per
  island (`task_sys_msg_island_suffixes`), text feedback on, meta
  recommendations off, and a plan-level proposal filter
  (`plan_precheck.py`: exact duplicates and strategy families that
  already have `SHINKA_PRECHECK_MAX_FAMILY` (3) valid plans are rejected
  before the evaluation) in place of the code-embedding novelty judge.
- `count_valid_plans.py` reports `n_families_valid` and
  `n_patch_cells_valid`; `diversity_report.py <run>` prints per-island
  and archive diversity and writes `diversity_by_gen.csv`.

**Level 2 — execution variants** of the successful plans, after a run:

```bash
python examples/loco_manipulation/sweep_to_variants.py \
    --run-dir results/<exp>/<scene>/<run> --top-families 5 --jobs 2
```

It groups successful plans by strategy, keeps the best plan(s) of each
family, and re-solves TO under one-factor-at-a-time variants: robot
start pose (`start`), execution speed (`timing`), cost weights
(`weights`), looser hard-goal tolerances (`goal`), and plan edits
(`plan`: one more / fewer step pair, mirrored hands). `--variants
objects` adds box start offsets, which changes the task layout. Variants
that change kinematics are KSO-checked first. Each solve writes a
`to_run.pkl` (replayable with `replay_to_run.py`) under
`<run>/to_variants/`, plus `results.jsonl` / `summary.csv`; reruns
resume. `--dry-run` lists the plan × variant matrix.

The IK wrapper (`ik_eval.py`) reads the scene from the active task
module and decides feasibility by solving the full plan, retrying up to
`SHINKA_KSO_ATTEMPTS` (default 3) times on fresh solvers because KSO
verdicts are noisy near the 1e-2 cutoff. Plans that fail every attempt
are diagnosed on a goal-free solver pool: a goal-miss check (every
contact works, only the target is missed), a bisection over prefix
lengths with `SHINKA_KSO_PREFIX_ATTEMPTS` (4) tries each to find the
first break and that prefix's own violations, and a repair probe
(`SHINKA_KSO_PROBES`, default 2) that inserts two walking steps before
the breaking mode / the first grasp and re-solves once. All of it is
written into the feedback in plain language (`violation_feedback.py`,
`plan_checks.py`, `repair_probe.py`). An acados solver is compiled for
one horizon length, so plans are padded to the next multiple of
`SHINKA_KSO_BUCKET` modes (default 10, `0` = exact lengths) and solved
on that bucket's solver, the results and violations mapped back to the
plan's own states. `SHINKA_KSO_PAD=start` (default) repeats the first
mode before the plan: the initial state is fixed, so the padded nodes
hold it at zero cost and the plan is solved as if unpadded; `end`
repeats the last mode after it, where each padded node carries the end
pose's distance-from-start cost once more — on layered_pallet gen_3
(72 modes) the end-padded 80-mode solve failed every attempt while the
exact-length control solved on the soft goal (30 min for the 3 hard +
1 soft attempts) and the start-padded 80-mode solve on the hard goal at
its second attempt (20 min). KSO is noisy either way: both re-solves
put the keyframes up to 1.0–1.1 m (median 0.17–0.18 m) from the search's
own solve of the same plan. Compiled solvers are cached under
`examples/loco_manipulation/solvers/` and reused by later evaluations
and runs, and concurrent evaluators take one file lock while compiling,
so only one acados build runs at a time (an 80-mode 4-box compile alone
was OOM-killed at 16 GB; the hard-goal one took 12 min). Without buckets
a 4-box search compiled 4–7 solvers per plan (3–14 min each) and its
diagnoses ran 20–73 min; two of five evaluations were killed during the
prefix walk with no record at all. Now the full-plan
attempts' violations are written as a provisional record before the
walk (`to_status: skipped`, diagnosis `pending`), and the goal-miss check
and the walk stop at the evaluation budget (`SHINKA_EVAL_BUDGET_S`,
1500 s) with the bracket found so far (`timed_out`). The maximum plan
length (`MAX_NUM_NODES`, default `50`) is
global; set `SHINKA_IK_MAX_NUM_NODES=N` to override. `SHINKA_TO_N_JOB=N`
(default 6; `run_search.sh` sets 1) sets the worker threads of
each TO solve — with the current moto build, 2 threads segfaulted on an
860-node problem that solves single-threaded.

Before each KSO solve, `kso_warm_start.py` replaces faro's initial
guess wherever the plan changes support level by at least 0.3 m. faro
starts every state at the initial standing pose and resolves the
collision pairs there too, so for a plan that climbs a table and steps
down its far side, every state near the table starts metres away with
collision constraints that describe the table as seen from the start.
The SQP then stops with the stepping foot 0.2 m above the floor, or walks
back down the near side and misses the goal. The seed walks the pelvis
in xy from the start over the edge of each upper face to the goal, at
standing height over its support level, with the upper leg folded at
each straddle state and the heading turned toward the upper surface; the
collision pairs are then re-resolved at the seed. On
`motiondisco.parkour_pick_place_1` this turns the floor -> table -> far
side crossing from infeasible on every attempt into a first-attempt
solve. A sole that still cannot reach a face more than 0.5 m below the
other sole is named in the feedback as a large step down. Plans without
a level change are untouched; `SHINKA_KSO_WARM_START=0` disables the
seed.

TO runs up to `SHINKA_TO_MAX_ITERS` outer iterations (default 300; it
was 100). The best `truck_unload_stack` plan stopped at 100 with a max
violation of 1.09 and converged after 188 (16.5 min on one moto thread).
`SHINKA_TO_WARM_START=1` (or `pose`, zero velocities) seeds TO from the
KSO solution (`to_warm_start.py`) instead of faro's guess of the initial
pose at every node; it is off by default because on that truck plan it
made TO worse (max violation 20-23 after 100 iterations).
`SHINKA_TO_PRESOLVE=N` overrides faro's 50 collision-free presolve
iterations. A scene whose TO goal is `off` lets TO set objects down
anywhere the contacts allow (on `truck_unload_stack`, 1.6-2.3 m from
the target on the plans tried). It stays off there for trajectory
diversity; `to="hard"` with a `to_tolerance` builds and converged on one
truck plan, while faro's `soft` TO goal does not build (std::bad_cast). Compiled TO code lives in one folder per scene
(`moto_gen/<scene>/`, goal-on problems in `moto_gen/<scene>_to<mode>/`): the node count and
contact schedule are runtime parameters, so every plan length reuses it, and moto recompiles
only functions whose code changed. A missing folder is seeded from the newest old
per-length `moto_gen/<scene>_<nodes>/` folder.

TO uses `to_solve.solve` rather than faro's `ToTask.solve`. faro re-linearises the
collisions after every SQP iteration (`update(1)`), which leaves moto's restoration
phase no iterations, so the first line-search failure aborted the solve with status 2.
`to_solve` gives restoration `SHINKA_TO_RESCUE` iterations (default 10, `0` aborts like faro's
loop, at most `SHINKA_TO_MAX_RESCUES`=20 times), cold-starts the interior point after
each rescue (`SHINKA_TO_RESET_AFTER_RESCUE=1`), records `to_fail_reason` and `to_rescues`
in `extra.json`, and stores a non-converged solve's lowest-residual iterate as
`results["best_iterate"]` in `to_run.pkl`.

Long plans can be solved in segments (`to_segments.py`). With
`SHINKA_TO_SEGMENT_MODES=N` (0 = off, the default) a plan longer than N modes is cut
into segments of at most N modes (and at least `SHINKA_TO_SEGMENT_MIN_MODES`, default 4),
solved in order, each starting from the previous segment's final state, and stitched into
one trajectory. faro fixes a TO's initial condition to node 0 of the warm start (robot
q, v, CoM, momentum; object pose and twist), so a segment continues the motion without
stopping. Cuts go only at double-support modes where no hand contact changes: a solve's
first interval is never a hand-switch interval, so a cut at a grasp or release would drop
the shortened, zero-wrench landing interval the full problem has there; foot landings and
object placements are exact. `walk()` steps have no double support, so a long walk has no
cut; a stretch like that longer than N makes only its own segment longer (the split
minimises the modes over N first). `SHINKA_TO_SEGMENT_CUTS=manip` cuts at the last step landing
before each grasp, release or placement (`to_segments.manipulation_cuts`), so a segment
starts with the robot at the object; `approach` cuts at the first landing after each
manipulation, `first` at its first mode (a hand cut); the default `balanced` picks the
fewest, most even segments. `SHINKA_TO_SEGMENT_LOOKAHEAD=L` solves each intermediate
segment L modes past its cut and keeps only the part up to the cut (`to_segments.windows`),
so its end state is planned for what follows; without it a segment's end is free and its
state cost pulls it back toward its own start. Use both: on a 43-mode climb_table_up_down
plan `manip` with L=5 converged every segment in 8 min, while without lookahead every cut
placement lost convergence partway through. The same setup solved that plan repeated
twice (85 modes, 1700 nodes, KSO-feasible): 12 segments, all converged, physically valid,
15 min. Other options: `SHINKA_TO_SEGMENT_GOAL=kso` ends intermediate
segments on hard base / object position targets from the KSO solution
(`SHINKA_TO_SEGMENT_GOAL_TOL`, 0.1 m), `SHINKA_TO_SEGMENT_REF=initial` keeps the full
problem's state-cost reference, and `SHINKA_TO_SEGMENT_POLISH=1` runs one full-horizon
solve warm-started from the stitched trajectory (`_POLISH_ITERS`, 50; `_POLISH_PRESOLVE`,
0). `extra.json` / `to_run.pkl` record one entry per segment in `to_segments` and the
polish in `to_polish`. Segments of any length share the scene's compiled TO code. The
polish is affordable from a good stitch: on three_shelf gen_25 (73 modes, 1460 nodes,
3 boxes) the full problem converged in 85 iterations from the stitched trajectory
(21 min single-threaded), TO cost 39.1 → 27.1, and on gen_27 (76 modes, one of 8
segments not converged but physically valid) in 77 iterations (35 min), cost 40.6 → 22.7.
It needs about 15 GB of memory (14 GB was not enough to read the results back).

A chained solve no longer runs every segment after one has gone wrong. A segment that
neither converges nor passes the physical check is first repaired when TO's own solution
of its window shows a foot travelling more than 0.8 m at a landing (`step_feedback`): the
walking steps the feedback would have asked for are inserted into the plan
(`plan_checks.insert_walk`, the later cuts shift) and the segment is re-solved from the
same start (`SHINKA_TO_REPAIR`, on; at most `SHINKA_TO_REPAIR_MAX` 2 repairs per segment,
`SHINKA_TO_REPAIR_TOTAL` 4 per plan, `SHINKA_TO_REPAIR_STEPS` 3 steps per repair — a badly
non-converged iterate is no measure of the distance: layered_pallet gen_6's stalled
segment, residual 8.4, asked for 8 steps at once; and a second repair before the same
mode is refused unless the first shortened the longest stride by 20 %, since gen_9's
four repairs at mode 32 left it at 1.5–2.1 m — the window failed for another reason);
with `SHINKA_TO_SEGMENT_MODES` set the rest of the
plan is cut afresh after an insertion, so segments do not grow with the repairs. On
layered_pallet gen_1 (41 modes, 2 of 8 segments converged in the search) six repairs
over four segments (plan 41 → 62 modes, foot travel 1.4–2.2 m before each) carried the
solve to 51 of 62 modes — segments 0–4 converged, 5 and 6 were physically valid — before
the last segment failed; it took 2.2 h against 45 min, the repaired windows having grown
to 18 modes (17 min a solve), hence the re-cut and the lower cap. A plan that short of
steps everywhere is better handed back to the LLM with the first repairs as feedback, and
the prompt now tells it to count steps from the geometry (`ceil(distance / 0.4) + 1`).
The repaired plan is the one the result describes:
`extra.json` holds it as `contact_mode_seq`, the proposed one as
`original_contact_mode_seq`, the edits in `plan_repairs`, and the feedback tells the LLM
which `walk(plan, k)` to add before which mode of its own plan; both plan hashes go to
`seen_plans.json`. If the segment still fails, `SHINKA_TO_SEGMENT_FAIL_FAST` (on) stops
there: the rest of the plan is held in place, the failing window's physical report (its
modes shifted to the full plan) becomes the result's, and the feedback names the segment
("in segment 3 of 8 (modes 8–17; modes 0–7 were solved)"). On layered_pallet gen_1 the
old behaviour spent 26 more minutes on five segments after segment 2 had failed. In
quality scoring such a plan gets `0.1 * (1 + 0.05 * solved_modes / n_modes)`
(0.100–0.105, below the weakest valid trajectory seen, 0.1085), so the search is pulled
toward plans that get further. `SHINKA_TO_STALL_ITERS=K` (0 = off) stops any TO solve
whose best residual improved by less than 10 % over the last K outer iterations instead
of spending its whole iteration budget — unless that residual is already within 3× the
feasibility tolerance, where a plateau often precedes convergence (pallet windows were
being cut at 0.147 against 0.1).

**Incremental evaluation** (`SHINKA_TO_INCREMENTAL`, on by default; `0` restores the
full-plan KSO and the global cut placement) is the one code path for any plan length. A
plan longer than the segment length is evaluated window by window from the start, and
nothing else runs on the whole plan until the end. On three_shelf gen_25 (73 modes) the
first pass converged all 8 windows (TO cost 39.1, the same as the whole chained solve;
3–7 min a window) and a second pass of the same plan took seven windows from the cache
and solved the last one, 4 min in all.

- The windows are cut left to right (`to_segments.greedy_cuts`): from each boundary the
  next one is the last step landing before a grasp, release or placement within the
  segment length (`manipulation_cuts`), else the last allowed cut (double support, no
  hand change) within it, else the first allowed cut beyond. A boundary therefore depends
  only on the plan up to the segment length past it, so plans that share a prefix share
  its windows; the global dynamic program of `split_points` could move an early cut
  because of a late change. A plan within the segment length is a single window.
- Each window is TO-solved from the state the previous window ended in. KSO runs only
  on the first window, from the task's own initial state (`to_eval._window_kso` →
  `ik_eval.check_plan`, a solver per window-length bucket; no full-plan KSO, no prefix
  bisection; `evaluate.py` skips its KSO stage), where its verdict is the full-plan
  KSO's for a short plan: a first window without a kinematic solution is a
  KSO-infeasible plan (diagnosis kind `window`). From a TO end state KSO is not a
  reliable gate — on three_shelf gen_25 it found no solution for windows 1 and 2, which
  TO then converged on — so later windows go straight to TO, whose violations and
  physical check explain a failure.
- Every converged window is stored in the run's cache (`segment_cache.py`,
  `<run>/segment_cache/`, one pickle per key, keyed by the plan up to the window's end,
  the start state's numbers and the solver settings). A later plan that keeps the prefix
  reuses those windows without solving them, so a 70-mode plan whose first 50 modes were
  solved before costs only the last 20. Failed windows are not stored (TO is not
  deterministic), nor are physically valid but non-converged ones: they count for their
  own plan, but their end state is not a start to build on (layered_pallet gen_32 reused
  one and the next window's presolve was infeasible at once). `ToSolve.cached_modes` /
  `extra.json`'s per-segment `cached` flag say what was reused.
- The first failing window stops the evaluation (fail-fast above), the score climbs with
  the solved modes, and the feedback tells the LLM which modes are solved and cached
  ("keep them exactly as they are and change the plan from mode 50 on"); the prompt's
  solver rules say the same. When every window passes, the stitched trajectory is
  checked physically and polished with the full-horizon TO as before.

`SHINKA_TO_SEGMENT_JUNCTION=static` (`to_static.py`) makes the segments independent
instead of chained: every cut gets a statically stable pose (faro's static IK with the
cut mode's contacts and object force balance, started from the KSO keyframe there and
anchored to it), and each segment runs from rest at its start pose to a hard terminal
target at its end pose (robot configuration and box poses within 2 cm / 0.05 rad,
terminal velocity cost ×1000). The poses and then the segments are solved by
`SHINKA_TO_SEGMENT_WORKERS` forked processes (faro's scene cannot be pickled, so
workers inherit the task and send back only results), so one failed segment no longer
spoils the ones after it; the stitched motion stops briefly at every junction, and with
`SHINKA_TO_SEGMENT_POLISH=1` the full-horizon polish runs afterwards to remove the stops.
It needs the KSO solution (`evaluate.py` passes it) and falls back to chaining without
it; a cut whose static pose fails or lands more than 0.3 m from its keyframe is dropped
(its segments merge). The junction targets have their own codegen folder
(`moto_gen/<scene>_tohard_segstatic`), seeded from the scene's goal-free folder so only
the goal functions compile. The junctions inherit KSO's choices, which TO may dislike,
and that is what limits the option so far. On three_shelf gen_27 (76 modes, 9 segments)
5 segments converged against 7 of 8 chained: the four failures sit where KSO, which has
no dynamics or stride limit, put the robot 0.2–0.7 m from where the chained TO had it
(at mode 72 the carried box was still 0.7 m from its shelf with four modes left).
Three workers cut the segment phase from 22 to about 14 min, each solve running
1.5–2× slower while three shared the machine. The polish cannot repair such a stitch:
from it the full problem ended at residual 33 after 100 iterations (42 min).

`SHINKA_TO_SEGMENT_JUNCTION=parallel` (`to_static.solve_parallel`) keeps the chained
formulation (free ends, lookahead windows) but solves the windows in parallel and repairs
the junctions afterwards: round 0 starts every window from rest at the static pose of its
cut; each later round re-solves, in parallel, only the windows whose start differs from
the previous window's end (base 5 cm, joints 0.1 rad, base velocity 0.1 m/s, joint
velocity 0.5 rad/s, objects 3 cm), from that end and warm-started with the window's own
previous solution, until every junction matches or as many rounds as windows have run
(then it has cost a sequential pass plus round 0). The result is the chained stitch;
`to_polish` records the rounds (`rounds`, `unmatched`). Measured on three_shelf gen_27
(8 windows, 2 workers): round 0 took 45 min of wall time (6 of 8 windows converged, each
solve 1.5–2× slower than alone) and every junction then mismatched by 0.18–0.92 m, so
round 1 re-solved 7 of 8 windows — the warm re-solves converged in 230–250 s against
380–690 s cold, but the whole thing is the chained pass (22 min, 7/8) plus round 0. No
gain on this machine; `chain` stays the default.

**Scene campaign.** `register_generated.py --examples <dir> --all` registers generated
scenes under `scenes/generated/` from their `spec.json` and `source/` files: `scene.py`,
`initial_conditions.py` and `spec.json` are copied unchanged and `goals.py` is rewritten
to MotionDisco's conventions (box poses within 5 cm and 0.1 rad of the generated goal as
a hard goal in KSO and in TO, the goal residual as scorer; the generated discrete goal is
kept), nothing else is read. Both solvers need the goal: with position-only goals the
first six campaign scenes were "solved" by plans that left the parcel up to 90° off its
target yaw (the quarter-turn scene by plans that never turned it), and with the TO goal
off, TO drifted to 70–80° off the yaw KSO had reached on 2 of 5 plans. With the hard TO
pose goal those plans converge and land the box within 6°; the goal variant of the TO
code compiles once per scene (~10 min, `moto_gen/<scene>_tohard`).
`scene_campaign.py --valid N --hours H` then runs one search per registered scene,
easiest first (boxes, then terrain and manipulation difficulty from the spec), with the
spec's description as the LLM-facing task text: TO code generated once per scene outside
the evaluations, the search under a systemd scope (`--cpus`, `--mem`), stopped at N unique
valid plans, resumed with `--extra` nodes when its budget runs out (up to `--cap`), given
up at H hours; `<out>/campaign_summary.jsonl` / `.csv` record every scene (outcome, valid
plans and their mode counts, evaluations, hours) and a rerun skips solved scenes. The leg
collision margin (`--leg-margin 0.02` on `--margin-capsules`) is on by default.

**Body/box clipping and the collision margin.** faro's collision constraints are
`distance >= 0` between the robot's capsules and the objects, with no margin, and the
G1's capsules are smaller than its meshes at the hands, elbows, ankles and hips; so valid
trajectories showed the mesh 1–3 cm inside a box at grasps, set-downs and when standing
next to a placed box (`to_physical` penetration up to 2.9 cm, under the 3 cm threshold
that was calibrated as mesh noise). The solver tolerance is not the lever — a window
solve at 0.05 had the same penetrations as at 0.1, to the millimetre, since they are
fixed by the contact geometry. `SHINKA_COLLISION_MARGIN=<m>` inflates the robot's
capsules by `m` (`_task_loader.with_collision_margin`; capsule sizes are runtime
parameters, no recompile; the cache key includes it). On the G1 it cannot be applied to
every capsule: even 1 cm makes the fixed standing start pose infeasible, because the
torso and arm capsules already sit against the thighs. `SHINKA_COLLISION_MARGIN_CAPSULES`
names the capsules it applies to: 2 cm on `l_shin,r_shin,l_foot,r_foot` removed gen_14's
ankle-in-box and elbow-in-box clips, converged 7 of 8 windows against 4 of 8 without it
(TO cost 57 against 199, 40 min). The feet cannot carry it on scenes where the robot
stands on a box: a planted sole is exempt from the floor plane but not from a box
support, and the foot capsule's bottom sits only 5 mm above the surface, so with 2 cm
every KSO of raised_dock_downward_transfer_001 failed at mode 0 ("foot penetrates
loading_dock by 1.5 cm"); the campaign uses the shins only. The remaining overlaps
(≤ 0.8 cm, wrist and shoulder meshes inside a box at grasps and releases) are the hand
geometry of the collision model and need a per-pair margin in faro.

A trajectory does not need TO convergence to count. moto's convergence test is on a
slack-inflated residual that mixes units, so physically fine trajectories were marked as
failed. `trajectory_check.py` judges the trajectory itself: centroidal dynamics,
torque-speed and joint limits, contact pull / friction / gap / slip, and penetration in a
MuJoCo replay. For a non-converged solve `to_eval` keeps the final or best iterate that
passes — after a numerical failure (status 2) only the best iterate is a candidate, since
the final one is the solver's wreck (after a presolve failure the held initial pose, which
a walking window passes trivially: shelf_unstack gen_22 was "physically valid" with
nothing moved), and such a window never counts as solved; scoring (quality mode), the LLM
feedback and dataset export use that verdict
(`to_valid`, `to_physical` in `extra.json`). `python trajectory_check.py <run_dir>`
prints the checks for every stored plan; `SHINKA_TO_PHYSICAL_CHECK=0` disables it.
The physical check knows nothing about the task goal, so `ToSolve.valid` also requires
the hard TO goal: `to_eval.goal_check` measures the final state against every
terminal-goal target with faro's own residual (`_target_error`: position, rotation log
in the target frame, joints) and the targets' componentwise `to_tolerance`, allowing
the feasibility tolerance as slack — the standard a converged solve is held to
(`to_goal` in `extra.json`, `ToSolve.goal_met`; a miss gets its own feedback block
naming the distance and angle to the target). On the stored runs both non-converged
valid plans pass it (end-standing gen_1: 4.8 cm, 0.112 rad against 5 cm / 0.1 rad).

`scene_campaign.py --scenes a b c` runs any registry scenes in the given order (the
MotionDisco scenes have no spec.json: the box count is read off `scene.py`, the description
comes from `run_search.sh`'s table), and `--to-goal hard` forces a hard TO goal with
a 5 cm / 0.1 rad tolerance on scenes whose `goals.py` leaves TO free (`SHINKA_TO_GOAL`,
`SHINKA_TO_GOAL_TOL`, `_task_loader.with_to_goal`; part of the segment-cache key and of the
codegen folder suffix), so that "valid" means the object reached its target as in the
generated campaign. Solved, capped and budget-exhausted scenes are skipped on a restart
unless `--redo`.

Incremental evaluation checks kinematics on the first window only, so a grasp the arms
cannot make reaches TO unexamined and fails there with wrists past their limits and the
box inside the torso — nothing the LLM can act on. On
occupied_side_resting_support_reorientation_001 (box 0.72 × 0.90 × 0.40 m) every plan of
the first hour gripped the 0.90 m left / right pair (or left + front with the front face
on the floor) and none converged. `grasp_probe.py` answers that with the solver: for the
failing window's grasp mode it asks faro's static IK (`to_static.static_pose`, a
statically stable pose with the mode's contacts, seeded with the trajectory's state at the
grasp and, failing that, with the window's start — the IK is anchored to its seed: from
the window's start, metres away, it found no pose even for pairs that have one, and a
failed window's iterate at the grasp can be garbage and fail just the same) whether a
standing pose exists; when none does, it probes the object's other opposite-face pairs
(closest together first, faces on a support excluded) and names one a pose exists for —
"no standing pose exists in which the robot holds parcel on its left and right faces
(0.90 m apart) … A standing pose does exist holding its top and bottom faces (0.40 m
apart): grip those instead." A grip on adjacent faces or one hand usually has a static
pose (friction holds the box) and `grip_feedback` already calls it weak, so for those
the probe goes straight to the opposite pairs and names the one the arms can span. The
robot model decides, not a span constant; it is feedback only
(`SHINKA_GRASP_PROBE=0` off, `SHINKA_GRASP_PROBES` static IK solves per evaluation,
default 3 at ~15–25 s each, only on failed windows).

Each evaluation runs under `job_config.time` (75 min in
`task/loco_manipulation.yaml` and `run_evo.py`); before TO starts the
evaluator writes a provisional "KSO feasible, TO pending" `metrics.json`,
so a plan whose TO is cut off by the limit still enters the run as a
KSO-feasible record instead of being lost.

**Prereq:** KSO needs acados (including its `bin/t_renderer` code
generator) and TO needs a `moto` build matching
`submodules/faro/extern/moto` (see faro's
`install.md`). Before running any of the commands below, export:

```bash
export ACADOS_SOURCE_DIR=/path/to/faro/extern/acados
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
export PYTHONPATH=submodules/faro/src:$PYTHONPATH
```

### Replaying a saved run

Each successful evaluation writes two pickles alongside
`extra.json` and `metrics.json`:

- `ik_run.pkl` — stage-2 KSO trajectory.
- `to_run.pkl` — stage-3 TO trajectory (only written when IK passes
  and TO builds successfully; may also carry a torque trajectory).

To replay either in a mujoco viewer:

```bash
# IK trajectory
PYTHONPATH=submodules/faro/src \
python examples/loco_manipulation/replay_ik_run.py \
    examples/loco_manipulation/results/.../ik_run.pkl

# TO trajectory (prints to_cost / to_converged and a torque summary
# when present)
PYTHONPATH=submodules/faro/src \
python examples/loco_manipulation/replay_to_run.py \
    examples/loco_manipulation/results/.../to_run.pkl
```

Shared flags:

- `--dt 0.5` — seconds between frames (default `0.3`)
- `--once`   — play the trajectory once instead of looping forever

Both scripts share the same viewer loop and use the active
`SHINKA_LOCOMANIP_{SCENE,IC}` env vars (defaults: `example_scene`
/ `default`) to pick which scene to build the MuJoCo model for
(`faro.configs.mujoco.build_mujoco`). Set them to match the task that
produced the pickle. Pickles written before the faro port (bare
`robot_xs` array) still replay.

Neither script loads acados, so you do **not** need
`ACADOS_SOURCE_DIR` / `LD_LIBRARY_PATH` set just to view a saved
run — only `PYTHONPATH=submodules/faro/src` is required.

## Data campaign: layouts, physics variants and an RL-tracker dataset

Three pieces turn searches into training data for a tracking policy.

**1. Layout variants (search level).** `layouts.py` registers named
layouts per scene: offsets for the robot start (`robot`), for fixed and
movable objects (`objects`, relative to the base IC's placements; objects
resting on a moved object move with it), and how the terminal goal moves
(`goal_follows: <object>` or `goal_shift`). Use them as IC names
`<base_ic>@<layout>`:

```bash
motiondisco_launch task=loco_manipulation variant=loco_diverse cluster=local \
    task.scene=motiondisco.long_dist_pick_place task.initial_condition=default@table_left
ICS="default default@near" SCENES=truck_unload_stack DIVERSE=1 SCORE_MODE=quality \
    ./examples/loco_manipulation/run_search.sh
```

The name works everywhere an IC name does (prompt, evaluator, replays,
post-search tools). `load_task` shifts world-position goals with the
layout, rebinds goal-based scorers, and rejects layouts in which fixed
boxes overlap or the robot starts inside one. Non-default ICs are appended
to the search script's `exp_name` (`.../<scene>/<ic>/<run>`).

**2. Physics-varied re-solves (`generate_dataset.py`).** For every plan of
a finished run whose TO converged or came within `--near-feasible`
(max violation, default 0.1), at most `--max-per-family` (3) per strategy
family: a reference solve plus `--samples` (8) random combined variants of
object mass (×0.5–2), object size (×0.85–1.15 per axis), robot start
(±0.15 m, ±0.2 rad), movable-object start (±0.08 m, ±0.15 rad),
`time_per_mode` (0.8–1.3 s) and TO cost weights (×0.5–2, log-uniform).
`--factors` picks the factors, `--range name=lo,hi` changes a range,
`--seed` fixes the sampling (seeds also depend on run, plan and index).
Friction is not varied: it is a constant (mu = 0.5) inside faro's TO.
Mass and size are runtime TO parameters (no recompilation); world-position
goals get their height corrected for size changes. Samples that change
geometry or start poses are KSO-checked first.

```bash
python examples/loco_manipulation/generate_dataset.py \
    --run-dir results/<exp>/<scene>/<run> --out datasets/loco --samples 8
python examples/loco_manipulation/generate_dataset.py --runs-root results/<exp> --dry-run
```

**3. Output.** Exported solves (converged, or `near_feasible`) land in
`<out>/<scene>/<ic>/<run>/<plan12>/<sample>/`:

- `traj.npz` — faro's `pin_to_mj` format at `--frequency` (50 Hz): root /
  joint / object positions, rotations (wxyz) and world-frame velocities,
  per-sample `contact_id`; plus the native solver nodes (`solver_time`,
  `solver_t_scale`, `solver_torques`, `solver_wrench_<patch>`,
  `solver_obj_xs_<obj>`, `solver_obj_wrench_<obj>`, `solver_robot_xs`) and
  `params_json`. Load with `faro.utils.results.load_trajectory_archive`.
- `scene.xml` — `save_scene_xml` of the varied scene (geometry, masses,
  start poses).
- `params.json` — scene / IC / layout, run and plan identity, the contact
  plan, per-object mass / inertia / half-extents / start pose, fixed
  objects, robot start q, mu, the sampled factors and seed, TO settings and
  result, git versions.

`<out>/manifest.jsonl` has one row per attempted sample (with the reason
when not exported) and `summary.csv` counts them per scene and IC. Reruns
skip samples already in the manifest.

**Campaigns.** `data_campaign.py campaigns/<name>.yaml` runs, for every
(scene, layout) in the file, one search (through `run_search.sh`,
which skips existing runs) and then `generate_dataset.py` on it; rerunning
resumes. `--dry-run` prints the matrix. See `campaigns/example.yaml`.

## Ingredients

- `env.py`: scene representation, contact_mode_seq type definitions, validation, milestone tracking.
- `initial.py`: seed solution; exposes `run_planning()`.
- `evaluate.py`: validator + scorer using `run_shinka_eval`.
- `run_evo.py`: async evolution runner.
- `shinka_small.yaml`: dev run profile (async 2/2/2 workers, 100 generations, $0.5 budget).

## Execution

From repo root:

```bash
cd examples/loco_manipulation
```

Async evolution:

```bash
python run_evo.py --config_path shinka_small.yaml
```

Single-program evaluation (no evolution loop):

```bash
python evaluate.py --program_path initial.py --results_dir results/manual_eval
```

## Running with local models (Ollama)

By default the example uses cloud LLMs configured in `shinka_small.yaml`
(or, on the Hydra path, in `shinka/configs/task/loco_manipulation.yaml`).
To run the entire pipeline against a local Ollama server instead — every
mutation, novelty-judge, meta-recommendation, prompt-evolution and
code-similarity-embedding call:

1. Start Ollama and pull the models you want to use:
   ```bash
   ollama serve                       # in another shell
   ollama pull qwen2.5-coder:14b      # mutation model
   ollama pull qwen2.5-coder:7b       # meta / novelty / prompt-evo
   ollama pull embeddinggemma         # OpenAI-compatible embeddings
   ```
2. Optional (defaults to `"local"` if unset):
   ```bash
   export LOCAL_OPENAI_API_KEY=local
   ```
3. Standalone runner:
   ```bash
   python run_evo.py --config_path shinka_ollama.yaml
   ```
   Hydra runner:
   ```bash
   motiondisco_launch task=loco_manipulation variant=ollama_local cluster=local
   ```

To go back to cloud models, drop the `variant=ollama_local` flag (or use
`shinka_small.yaml`). Both presets share the same `db_config` /
`evo_config` skeleton; only the LLM and embedding fields differ.

Substitute the model IDs in `shinka_ollama.yaml` and
`shinka/configs/variant/ollama_local.yaml` for whatever you actually have
pulled (`ollama list`). Mixing cloud and local strings in the same
`llm_models` list is allowed — UCB selection will pick between them, so
you can keep a strong cloud model as a fallback while iterating cheaply
on local ones.

See `docs/support_local_models.md` for the full inline-URL format and
the list of consumers. Note: structured-output mode is not yet supported
for `local/...` models.
