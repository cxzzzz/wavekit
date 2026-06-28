#!/usr/bin/env bash
# In-container VCS build script for the unified comparison FSDB fixtures.
# Generates two FSDB files:
#   compare.fsdb     — all signal types, two-state values
#   compare_xz.fsdb  — X/Z states for unknown mask testing
# Invoked by build_compare.local.sh; runs inside the Synopsys Docker image.
set -euo pipefail

fixture_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/tests/readers/fixtures"
out_dir="${1:-$fixture_dir/fsdb}"
work_dir="${WAVEKIT_COMPARE_FSDB_BUILD_DIR:-/tmp/wavekit_compare_fsdb_build}"

mkdir -p "$out_dir" "$work_dir"

# --- compare.fsdb (all signal types, no X/Z) ---
echo "== Building compare.fsdb with VCS =="
vcs -full64 -sverilog \
  -debug_access+all \
  +define+COMPARE_VCS \
  -o "$work_dir/simv_compare" \
  -l "$work_dir/vcs_compare_compile.log" \
  "$fixture_dir/compare_unit.sv" \
  "$fixture_dir/compare_dut.sv" \
  "$fixture_dir/compare_tb.sv"

"$work_dir/simv_compare" +fsdbfile="$out_dir/compare.fsdb" -l "$work_dir/vcs_compare_run.log"
echo "Wrote $out_dir/compare.fsdb"

# --- compare_xz.fsdb (X/Z states) ---
echo "== Building compare_xz.fsdb with VCS =="
vcs -full64 -sverilog \
  -debug_access+all \
  +define+COMPARE_VCS \
  -o "$work_dir/simv_xz" \
  -l "$work_dir/vcs_xz_compile.log" \
  "$fixture_dir/compare_xz_tb.sv"

"$work_dir/simv_xz" +fsdbfile="$out_dir/compare_xz.fsdb" -l "$work_dir/vcs_xz_run.log"
echo "Wrote $out_dir/compare_xz.fsdb"
