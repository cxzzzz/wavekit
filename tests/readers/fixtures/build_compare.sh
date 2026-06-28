#!/usr/bin/env bash
# Build the unified comparison VCD and FST fixtures.
# Generates 4 files:
#   vcd/compare.vcd, fst/compare.fst       — Verilator (all signal types, X only)
#   vcd/compare_xz.vcd, fst/compare_xz.fst — Icarus (X/Z states, simple signals)
#
# Usage:
#   ./build_compare.sh
set -euo pipefail

fixture_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
vcd_out="$fixture_dir/vcd/compare.vcd"
fst_out="$fixture_dir/fst/compare.fst"
xz_vcd_out="$fixture_dir/vcd/compare_xz.vcd"
xz_fst_out="$fixture_dir/fst/compare_xz.fst"
work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

# --- Verilator (all signal types, two-state, X treated as 0) ---
echo "== Compiling compare_tb with Verilator =="
verilator --binary --timing --trace --trace-structs \
  -Wno-BADVLTPRAGMA -Wno-WIDTHTRUNC -Wno-WIDTH \
  -Mdir "$work_dir/vbuild" \
  -o "$work_dir/vbuild/Vcompare_tb" \
  "$fixture_dir/compare_unit.sv" \
  "$fixture_dir/compare_dut.sv" \
  "$fixture_dir/compare_tb.sv"

echo "== Running Verilator simulation =="
mkdir -p "$(dirname "$vcd_out")"
( cd "$work_dir/vbuild" && "$work_dir/vbuild/Vcompare_tb" )
mv "$work_dir/vbuild/compare.vcd" "$vcd_out"
echo "Wrote $vcd_out"

echo "== Converting Verilator VCD -> FST =="
mkdir -p "$(dirname "$fst_out")"
vcd2fst "$vcd_out" "$fst_out"
echo "Wrote $fst_out"

# --- Icarus Verilog (X/Z states, four-state) ---
echo "== Compiling compare_xz_tb with Icarus Verilog =="
iverilog -g2012 -o "$work_dir/compare_xz.vvp" "$fixture_dir/compare_xz_tb.sv"

echo "== Running Icarus simulation =="
mkdir -p "$(dirname "$xz_vcd_out")"
( cd "$work_dir" && vvp "$work_dir/compare_xz.vvp" )
mv "$work_dir/compare_xz.vcd" "$xz_vcd_out"
echo "Wrote $xz_vcd_out"

echo "== Converting Icarus VCD -> FST =="
mkdir -p "$(dirname "$xz_fst_out")"
vcd2fst "$xz_vcd_out" "$xz_fst_out"
echo "Wrote $xz_fst_out"
