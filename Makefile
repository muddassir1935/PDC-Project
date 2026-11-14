# Parallel K-Means (Hybrid MPI + OpenMP) — build targets
#
#   make            build whichever implementations exist so far
#   make serial     bin/kmeans_serial    (M1: serial baseline)
#   make mpi        bin/kmeans_mpi       (M2: MPI version)
#   make omp        bin/kmeans_omp       (M3: OpenMP version)
#   make hybrid     bin/kmeans_hybrid    (M4: MPI + OpenMP)
#   make clean      remove build outputs

CC       := gcc
MPICC    := mpicc
CFLAGS   := -O2 -Wall -Wextra -std=c11
OMPFLAGS := -fopenmp
LDFLAGS  := -lm
BINDIR   := bin

SERIAL_SRC := src/kmeans_serial.c
MPI_SRC    := src/kmeans_mpi.c
OMP_SRC    := src/kmeans_omp.c
HYBRID_SRC := src/kmeans_hybrid.c

.PHONY: all serial mpi omp hybrid hybrid-static clean help

all:
	@built=0; \
	$(MAKE) --no-print-directory -s _build_one SRC=$(SERIAL_SRC) OUT=$(BINDIR)/kmeans_serial MODE=serial 2>/dev/null && built=1; \
	$(MAKE) --no-print-directory -s _build_one SRC=$(MPI_SRC)    OUT=$(BINDIR)/kmeans_mpi    MODE=mpi 2>/dev/null && built=1; \
	$(MAKE) --no-print-directory -s _build_one SRC=$(OMP_SRC)    OUT=$(BINDIR)/kmeans_omp    MODE=omp 2>/dev/null && built=1; \
	$(MAKE) --no-print-directory -s _build_one SRC=$(HYBRID_SRC) OUT=$(BINDIR)/kmeans_hybrid MODE=hybrid 2>/dev/null && built=1; \
	if [ $$built -eq 0 ]; then echo "No implementations yet — sources land in src/ from Milestone 1 onwards."; fi

serial:
	@$(MAKE) --no-print-directory -s _build_one SRC=$(SERIAL_SRC) OUT=$(BINDIR)/kmeans_serial MODE=serial 2>/dev/null \
		|| echo "kmeans_serial: not implemented yet (Milestone 1)."

mpi:
	@$(MAKE) --no-print-directory -s _build_one SRC=$(MPI_SRC) OUT=$(BINDIR)/kmeans_mpi MODE=mpi 2>/dev/null \
		|| echo "kmeans_mpi: not implemented yet (Milestone 2)."

omp:
	@$(MAKE) --no-print-directory -s _build_one SRC=$(OMP_SRC) OUT=$(BINDIR)/kmeans_omp MODE=omp 2>/dev/null \
		|| echo "kmeans_omp: not implemented yet (Milestone 3)."

hybrid:
	@$(MAKE) --no-print-directory -s _build_one SRC=$(HYBRID_SRC) OUT=$(BINDIR)/kmeans_hybrid MODE=hybrid 2>/dev/null \
		|| echo "kmeans_hybrid: not implemented yet (Milestone 4)."

# M5 benchmark variant only: same hybrid source with schedule(static),
# so the static-vs-dynamic comparison measures the schedule, not a
# different code path. Default hybrid build stays schedule(dynamic, 1024).
hybrid-static:
	@if [ -f "$(HYBRID_SRC)" ]; then \
		mkdir -p $(BINDIR); \
		$(MPICC) $(CFLAGS) $(OMPFLAGS) -DHYBRID_STATIC_SCHEDULE -o $(BINDIR)/kmeans_hybrid_static $(HYBRID_SRC) $(LDFLAGS) \
		&& echo "built $(BINDIR)/kmeans_hybrid_static"; \
	else \
		echo "kmeans_hybrid: not implemented yet (Milestone 4)."; \
	fi

# Internal: build one implementation if its source exists.
# MODE selects the compiler/flags for that implementation.
_build_one:
	@if [ -f "$(SRC)" ]; then \
		mkdir -p $(BINDIR); \
		case "$(MODE)" in \
			serial) $(CC) $(CFLAGS) -o $(OUT) $(SRC) $(LDFLAGS) ;; \
			mpi)    $(MPICC) $(CFLAGS) -o $(OUT) $(SRC) $(LDFLAGS) ;; \
			omp)    $(CC) $(CFLAGS) $(OMPFLAGS) -o $(OUT) $(SRC) $(LDFLAGS) ;; \
			hybrid) $(MPICC) $(CFLAGS) $(OMPFLAGS) -o $(OUT) $(SRC) $(LDFLAGS) ;; \
		esac && echo "built $(OUT)"; \
	else \
		exit 1; \
	fi

clean:
	rm -rf $(BINDIR)

help:
	@echo "Targets: all | serial | mpi | omp | hybrid | clean"
	@echo "  serial  -> $(SERIAL_SRC)  (plain gcc)"
	@echo "  mpi     -> $(MPI_SRC)     (mpicc)"
	@echo "  omp     -> $(OMP_SRC)     (gcc -fopenmp)"
	@echo "  hybrid  -> $(HYBRID_SRC)  (mpicc -fopenmp)"
