#!/usr/bin/env bash
# Run a MotionDisco search on one or more loco-manipulation scenes.
#
# Each scene gets one search (LLM proposals + KSO + TO + text feedback)
# through motiondisco_launch. Runs that already wrote to
# $RESULTS_DIR/<exp_name>/ are skipped, so a Ctrl-C + re-run resumes where
# the previous invocation stopped.
#
# Examples:
#   SCENES="motiondisco.box_stacking" ./examples/loco_manipulation/run_search.sh
#
#   # Diverse, quality-scored search, three independent repeats:
#   SCENES="motiondisco.box_stacking" DIVERSE=1 SCORE_MODE=quality REPEATS=3 \
#       ./examples/loco_manipulation/run_search.sh
#
#   # Continue an existing run for 30 more nodes:
#   SCENES="motiondisco.box_stacking" RESUME=1 RESUME_EXTRA=30 \
#       ./examples/loco_manipulation/run_search.sh
#
# Env-var knobs:
#   SCENES         space-separated scene IDs (required).
#   DESCRIPTION / MAX_NODES
#                  task sentence and node budget; default: the scene's
#                  entry in SCENE_DESCRIPTIONS / SCENE_MAX_NODES below.
#   DIVERSE=1      variant=loco_diverse (4 isolated islands, plan-patch
#                  descriptor archive, model ensemble, per-island strategy
#                  hints, task.diversity); exp_name gains "_diverse".
#   SCORE_MODE     legacy (default) | quality -- task.score_mode, see
#                  scoring.py; quality adds "_quality" to exp_name.
#   ICS="default default@far ..."
#                  initial conditions per scene; "<ic>@<layout>" applies a
#                  layout from layouts.py and is appended to exp_name.
#   REPEATS=N      independent runs per scene, exp_name suffixed _rep1..N.
#   TAG=<name>     appended to exp_name so a new campaign does not skip or
#                  resume older runs.
#   RESULTS_DIR    run root (default: results).
#   MAX_EVAL_JOBS  evaluations at the same time (max_evaluation_jobs).
#   JOB_TIME       wall-clock limit per evaluation, HH:MM:SS (job_config.time).
#   RESUME=1       continue the latest run dir of each scene instead of
#                  skipping it; RESUME_EXTRA=N runs N more nodes,
#                  RESUME_NODES=N sets the absolute node cap (counted from
#                  generation 0, so it must exceed the existing count).
#
# The node budget is evo_config.num_generations: the runner counts rows in
# programs.sqlite minus the gen-0 copy on each extra island, so N nodes
# means at most N-1 LLM proposals.
#
# Prerequisites: the motiondisco conda env (see the top-level README), the
# `claude` CLI logged in (the default model uses the claude-max/ backend;
# replace it with a bare model name to use an API key), and an Ollama
# embedding server at http://localhost:11434/v1.

set -u

export SHINKA_IK_MAX_NUM_NODES="${SHINKA_IK_MAX_NUM_NODES:-50}"
# Moto worker threads per TO solve; multi-threaded solves segfaulted on
# long plans with the current moto build.
export SHINKA_TO_N_JOB="${SHINKA_TO_N_JOB:-1}"

# motiondisco_launch is a console script, so the repo root must be on
# PYTHONPATH for the ${py:examples...} resolver in task/loco_manipulation.yaml.
# faro's editable install already provides submodules/faro/src.
_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="${_REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export ACADOS_SOURCE_DIR="${ACADOS_SOURCE_DIR:-$_REPO_ROOT/submodules/faro/extern/acados}"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RESUME="${RESUME:-}"
RESUME_NODES="${RESUME_NODES:-}"
RESUME_EXTRA="${RESUME_EXTRA:-}"
DIVERSE="${DIVERSE:-}"
SCORE_MODE="${SCORE_MODE:-legacy}"
REPEATS="${REPEATS:-1}"
TAG="${TAG:-}"
RESULTS_DIR="${RESULTS_DIR:-results}"
MAX_EVAL_JOBS="${MAX_EVAL_JOBS:-}"
JOB_TIME="${JOB_TIME:-}"
read -r -a ICS_ARR <<< "${ICS:-default}"
read -r -a SCENES_ARR <<< "${SCENES:-}"
case "$SCORE_MODE" in
    legacy|quality) ;;
    *) echo "SCORE_MODE must be legacy or quality, got '$SCORE_MODE'" >&2; exit 2 ;;
esac
if [ "${#SCENES_ARR[@]}" -eq 0 ]; then
    echo "Set SCENES, e.g. SCENES=motiondisco.box_stacking $0" >&2
    exit 2
fi

# Gen 0 is copied once per extra island (both island presets use 4
# islands); the runner subtracts these rows from its node count.
ISLAND_COPIES=3

declare -A SCENE_DESCRIPTIONS=(
    [motiondisco.banana_real]="Reach and grasp the banana."
    [motiondisco.box_stacking]="Stack the boxes as high as possible on the table."
    [motiondisco.climb_table_with_box]="Climb onto the table and hold the white box at the end."
    [motiondisco.climb_table_up_down_with_box]="Place the box on the other side of the table."
    [motiondisco.long_dist_pick_place]="Pick the yellow box and place it on the distant table."
    [motiondisco.move_through_clutter]="Navigate through the cluttered environment to the far side."
    [motiondisco.parkour_pick_place_1]="Pick the white box and place it on the elevated table."
    [motiondisco.parkour_pick_place_2]="Pick the white box and place it on the elevated table."
    [motiondisco.under_table_pick_place]="Pick the white box from under the table and place it on top."
    [climb_table_up_down_with_box]="Carry the white box over the table and set it down on the floor on the other side of the table."
    [floor_stack_carry]="Carry the two stacked boxes to the target spot and set them down there, still stacked."
    [truck_unload_stack]="Unload both boxes from the truck bed and stack them on the floor, the smaller box on top of the larger one."
)

# evo_config.num_generations per scene (= tree-node budget).
declare -A SCENE_MAX_NODES=(
    [motiondisco.banana_real]=27
    [motiondisco.box_stacking]=11
    [motiondisco.climb_table_with_box]=13
    [motiondisco.climb_table_up_down_with_box]=21
    [motiondisco.long_dist_pick_place]=20
    [motiondisco.move_through_clutter]=15
    [motiondisco.parkour_pick_place_1]=40
    [motiondisco.parkour_pick_place_2]=34
    [motiondisco.under_table_pick_place]=14
    [climb_table_up_down_with_box]=40
    [floor_stack_carry]=40
    [truck_unload_stack]=100
)
DEFAULT_MAX_NODES=40

# Hydra overrides for every run. The node budget is the only stopping
# criterion (max_api_costs stays unset; never set it to 0).
BASE_OVERRIDES=(
    database=island_small
    evolution=small_budget
    cluster=local
    "evo_config.llm_models=[claude-max/claude-opus-5-5]"
    "++evo_config.llm_kwargs.reasoning_efforts=[high]"
    "++evo_config.llm_kwargs.max_tokens=32768"
    "evo_config.embedding_model=local/embeddinggemma@http://localhost:11434/v1"
    "++evo_config.evolve_prompts=false"
    "++evo_config.meta_rec_interval=null"
    "++evo_config.use_text_feedback=true"
)

run_exists() {
    [ -d "$RESULTS_DIR/$1" ] && [ -n "$(ls -A "$RESULTS_DIR/$1" 2>/dev/null)" ]
}

# Latest timestamped run dir under $RESULTS_DIR/<exp_name>/.
find_existing_run_dir() {
    local entry
    for entry in $(ls -1 "$RESULTS_DIR/$1" 2>/dev/null | sort -r); do
        if [ -d "$RESULTS_DIR/$1/$entry" ]; then
            echo "$entry"
            return 0
        fi
    done
    return 1
}

# Nodes already in a run dir, in the runner's own budget units.
existing_node_count() {
    python3 - "$1" "$ISLAND_COPIES" <<'PY'
import glob, os, sqlite3, sys

run_dir, island_copies = sys.argv[1], int(sys.argv[2])
for path in sorted(glob.glob(os.path.join(run_dir, "*.sqlite"))):
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        n = con.execute("SELECT COUNT(*) FROM programs").fetchone()[0]
        con.close()
        print(max(0, n - island_copies))
        break
    except Exception:
        continue
else:
    print(0)
PY
}

# Overrides that variant=loco_diverse sets itself (database, models,
# efforts, meta recommendations); Hydra rejects them twice.
diverse_filter() {
    local o
    for o in "$@"; do
        case "$o" in
            database=*|evo_config.llm_models=*|*evo_config.llm_kwargs.*|*evo_config.meta_rec_interval=*) ;;
            *) printf '%s\n' "$o" ;;
        esac
    done
}

launch() {
    local scene=$1 desc=$2 rep=$3 ic=$4
    local -a overrides=("${BASE_OVERRIDES[@]}")
    local suffix=""
    if [ -n "$DIVERSE" ]; then
        mapfile -t overrides < <(diverse_filter "${overrides[@]}")
        overrides+=("variant=loco_diverse")
        suffix+="_diverse"
    fi
    overrides+=("task.score_mode=$SCORE_MODE")
    [ -n "$MAX_EVAL_JOBS" ] && overrides+=("max_evaluation_jobs=$MAX_EVAL_JOBS")
    [ -n "$JOB_TIME" ] && overrides+=("job_config.time='$JOB_TIME'")
    [ "$SCORE_MODE" != legacy ] && suffix+="_$SCORE_MODE"
    [ -n "$TAG" ] && suffix+="_$TAG"
    [ -n "$rep" ] && suffix+="_rep$rep"
    local nodes="${MAX_NODES:-${SCENE_MAX_NODES[$scene]:-$DEFAULT_MAX_NODES}}"
    # Directory name kept from earlier releases so existing runs resume.
    local exp_name="loco_abl3_ik_fb$suffix/$scene"
    [ "$ic" != default ] && exp_name+="/$ic"
    local -a out=()

    if [ -n "$RESUME" ]; then
        local existing have
        if ! existing=$(find_existing_run_dir "$exp_name"); then
            echo "[skip-resume] $exp_name (no prior run dir)"
            return 0
        fi
        have=$(existing_node_count "$RESULTS_DIR/$exp_name/$existing")
        if [ -n "$RESUME_EXTRA" ]; then
            nodes=$(( have + RESUME_EXTRA ))
        elif [ -n "$RESUME_NODES" ]; then
            nodes=$RESUME_NODES
        fi
        if [ "$nodes" -le "$have" ]; then
            echo "[skip-resume] $exp_name (cap $nodes <= existing $have nodes; use RESUME_EXTRA)"
            return 0
        fi
        echo "[resume] $exp_name (from $existing, have=$have, cap=$nodes nodes)"
        out=("output_dir=$RESULTS_DIR/$exp_name/$existing")
    elif run_exists "$exp_name"; then
        echo "[skip] $exp_name"
        return 0
    else
        echo "[run]  $exp_name (nodes=$nodes)"
    fi

    motiondisco_launch \
        task=loco_manipulation \
        "task.scene=$scene" \
        "task.initial_condition=$ic" \
        "task.description='$desc'" \
        "${overrides[@]}" \
        "evo_config.num_generations=$nodes" \
        "exp_name=$exp_name" \
        "results_dir=$RESULTS_DIR" \
        "${out[@]}" \
        || echo "[fail] $exp_name (continuing)"
}

for scene in "${SCENES_ARR[@]}"; do
    desc="${DESCRIPTION:-${SCENE_DESCRIPTIONS[$scene]:-}}"
    if [ -z "$desc" ]; then
        echo "[skip-scene] $scene (no description; set DESCRIPTION or add it to SCENE_DESCRIPTIONS)"
        continue
    fi
    for ic in "${ICS_ARR[@]}"; do
        if [ "$REPEATS" -le 1 ]; then
            launch "$scene" "$desc" "" "$ic"
        else
            for rep in $(seq 1 "$REPEATS"); do
                launch "$scene" "$desc" "$rep" "$ic"
            done
        fi
    done
done
