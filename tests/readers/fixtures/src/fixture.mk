# Shared build rules for a single fixture's SystemVerilog sources.
# Included by each fixtures/src/<name>/Makefile, which sets TB, SOURCES,
# TOOLCHAIN (optional), and FSDB_DEFS (optional) before including this
# file, then declares its own `all:` naming the targets it needs — not
# every fixture needs the same outputs (vcd, fst, fsdb, or some mix).
#
# Variables a fixture Makefile sets:
#   TB         - base name of the testbench module, e.g. compare
#   SOURCES    - explicit list of .sv files to compile, testbench last
#   TOOLCHAIN  - verilator or iverilog (selects which rule builds $(TB).vcd)
#   FSDB_DEFS  - optional vcs +define+... flags, forwarded to the fsdb target
#
# Targets this file provides (a fixture's `all` picks the ones it needs):
#   vcd   - generated/$(TB).vcd, via $(TOOLCHAIN); requires TOOLCHAIN to be set
#   fst   - generated/$(TB).fst, derived from the vcd via vcd2fst (lossless)
#   fsdb  - generated/$(TB).fsdb, via build_fsdb.local.sh (needs a VCS
#           environment; see that script's header). Silently skipped if
#           that script is absent (e.g. outside this machine).
#   clean - remove local build artifacts

FIXTURE_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
GENERATED_DIR := $(FIXTURE_ROOT)/generated

.PHONY: vcd fst fsdb clean

vcd: $(GENERATED_DIR)/$(TB).vcd
fst: $(GENERATED_DIR)/$(TB).fst
fsdb: $(GENERATED_DIR)/$(TB).fsdb

ifeq ($(TOOLCHAIN),verilator)
$(GENERATED_DIR)/$(TB).vcd: $(SOURCES)
	verilator --binary --timing --trace --trace-structs \
	  -Wno-BADVLTPRAGMA -Wno-WIDTHTRUNC -Wno-WIDTH \
	  -Mdir obj_dir -o V$(TB) $(SOURCES)
	( cd obj_dir && ./V$(TB) )
	mkdir -p $(GENERATED_DIR)
	mv obj_dir/$(TB).vcd $@
else ifeq ($(TOOLCHAIN),iverilog)
$(GENERATED_DIR)/$(TB).vcd: $(SOURCES)
	iverilog -g2012 -o $(TB).vvp $(SOURCES)
	vvp $(TB).vvp
	mkdir -p $(GENERATED_DIR)
	mv $(TB).vcd $@
else ifneq ($(TOOLCHAIN),)
$(error TOOLCHAIN must be verilator or iverilog, got "$(TOOLCHAIN)")
endif

$(GENERATED_DIR)/$(TB).fst: $(GENERATED_DIR)/$(TB).vcd
	mkdir -p $(GENERATED_DIR)
	vcd2fst $< $@

$(GENERATED_DIR)/$(TB).fsdb: $(SOURCES)
ifneq ($(wildcard ../build_fsdb.local.sh),)
	mkdir -p $(GENERATED_DIR)
	../build_fsdb.local.sh "$(TB)" "$@" $(FSDB_DEFS) -- $(SOURCES)
else
	@echo "$(TB): ../build_fsdb.local.sh not found, skipping fsdb"
endif

clean:
	rm -rf obj_dir *.vvp *.vcd simv_$(TB) simv_$(TB).daidir csrc
