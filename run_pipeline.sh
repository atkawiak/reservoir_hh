#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# run_pipeline.sh — Full reservoir regime pipeline (Etap A → B → Walidacja)
# ═══════════════════════════════════════════════════════════════════════════
#
# Usage:
#   ./run_pipeline.sh                          # seed=100, full pipeline
#   ./run_pipeline.sh --seed 42                # different seed
#   ./run_pipeline.sh --seed 100 --step AB     # only Etap A + B
#   ./run_pipeline.sh --seed 100 --step V      # only validation (D,E,F)
#   ./run_pipeline.sh --seed 100 --step ABVD   # A + B + validation + diagnostics
#   ./run_pipeline.sh --preset rc              # use RC preset for calibration
#
# Steps:
#   A = Etap A (regime_builder)
#   B = Etap B (regime_calibrator)
#   V = Validation tests (acceptance_tests D,E,F)
#   T = Acceptance tests A,B,C (separation, crosscheck, report)
#   D = Diagnostics (lambda_scan + rho_lambda_scan)
#
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Config ──
SEED=100
STEPS="ABTVD"
PRESET="rc"
PYTHON="/home/andrzej.kawiak/anaconda3/envs/scientificProject/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)   SEED="$2";   shift 2 ;;
        --step)   STEPS="$2";  shift 2 ;;
        --preset) PRESET="$2"; shift 2 ;;
        --python) PYTHON="$2"; shift 2 ;;
        -h|--help)
            head -20 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "═══════════════════════════════════════════════════════════════"
echo "  RESERVOIR REGIME PIPELINE"
echo "  seed=$SEED  steps=$STEPS  preset=$PRESET"
echo "  python=$PYTHON"
echo "═══════════════════════════════════════════════════════════════"
echo ""

T_START=$SECONDS

# ── Etap A: Build regimes ──
if [[ "$STEPS" == *A* ]]; then
    echo "▶ ETAP A: Building reservoir topologies (regime_builder.py)..."
    echo "────────────────────────────────────────────────────────────"
    "$PYTHON" -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from regime_builder import run
run(seed=$SEED)
"
    echo ""
    echo "✓ Etap A done.  NPZ files in: $SCRIPT_DIR/regimes/"
    echo ""
fi

# ── Etap B: Calibrate regimes ──
if [[ "$STEPS" == *B* ]]; then
    echo "▶ ETAP B: Calibrating regimes (regime_calibrator.py)..."
    echo "────────────────────────────────────────────────────────────"
    "$PYTHON" "$SCRIPT_DIR/regime_calibrator.py" \
        --seed "$SEED" --preset "$PRESET"
    echo ""
    echo "✓ Etap B done.  Calibrated NPZ in: $SCRIPT_DIR/regimes_calibrated/"
    echo ""
fi

# ── Acceptance tests A,B,C ──
if [[ "$STEPS" == *T* ]]; then
    echo "▶ ACCEPTANCE TESTS A,B,C (separation, crosscheck, report)..."
    echo "────────────────────────────────────────────────────────────"
    "$PYTHON" "$SCRIPT_DIR/acceptance_tests.py" \
        --seed "$SEED" --test ABC
    echo ""
    echo "✓ Acceptance tests A,B,C done."
    echo ""
fi

# ── Validation tests D,E,F ──
if [[ "$STEPS" == *V* ]]; then
    echo "▶ VALIDATION TESTS D,E,F (dt-stability, eps-invariance, time-convergence)..."
    echo "────────────────────────────────────────────────────────────"
    "$PYTHON" "$SCRIPT_DIR/acceptance_tests.py" \
        --seed "$SEED" --test DEF
    echo ""
    echo "✓ Validation tests D,E,F done."
    echo ""
fi

# ── Diagnostics ──
if [[ "$STEPS" == *D* ]]; then
    echo "▶ DIAGNOSTICS: Lambda scan + Rho-Lambda scan..."
    echo "────────────────────────────────────────────────────────────"

    echo "  → lambda_scan.py (full scan)..."
    "$PYTHON" "$SCRIPT_DIR/lambda_scan.py"
    echo ""

    echo "  → rho_lambda_scan.py..."
    "$PYTHON" "$SCRIPT_DIR/rho_lambda_scan.py"
    echo ""

    echo "✓ Diagnostics done."
    echo ""
fi

# ── Summary ──
ELAPSED=$(( SECONDS - T_START ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))
echo "═══════════════════════════════════════════════════════════════"
echo "  PIPELINE COMPLETE"
echo "  seed=$SEED  steps=$STEPS  time=${MINS}m${SECS}s"
echo "═══════════════════════════════════════════════════════════════"
