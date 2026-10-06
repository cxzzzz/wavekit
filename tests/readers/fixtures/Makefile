# Rebuild every fixture's declared outputs. Each src/<name>/Makefile's
# own `all` decides what it needs (vcd/fst, fsdb, or both) — fsdb targets
# call build_fsdb.local.sh and only succeed inside the VCS/Docker
# environment.
#
# Usage:
#   make            # build every fixture's declared outputs
#   make clean      # clean every fixture's local build artifacts

FIXTURES := $(patsubst src/%/,%,$(wildcard src/*/))

.PHONY: all clean $(FIXTURES)

all: $(FIXTURES)

$(FIXTURES):
	$(MAKE) -C src/$@ all

clean:
	for f in $(FIXTURES); do $(MAKE) -C src/$$f clean; done
