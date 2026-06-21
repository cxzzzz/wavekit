#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
fixture_dir="$repo_root/tests/readers/fixtures/fsdb"
out_file="${1:-$fixture_dir/simple.fsdb}"
work_dir="${WAVEKIT_FSDB_BUILD_DIR:-$repo_root/tests/fsdb_work}"
simv="$work_dir/simv"

mkdir -p "$(dirname "$out_file")" "$work_dir"
rm -f "$out_file" "$work_dir/vcs_compile.log" "$work_dir/vcs_run.log"
rm -rf "$simv" "$simv.daidir" "$work_dir/csrc"

vcs -full64 -sverilog \
  -debug_access+all \
  -o "$simv" \
  -l "$work_dir/vcs_compile.log" \
  "$fixture_dir/simple_dut.sv" \
  "$fixture_dir/simple_tb.sv"

"$simv" +fsdbfile="$out_file" -l "$work_dir/vcs_run.log"

printf 'Wrote %s\n' "$out_file"
