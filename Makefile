OUTPUTDIR := bin/
CFLAGS := -std=c++14 -fvisibility=hidden -lpthread #-Wall -Wextra

# Define different compilers for different targets
CXX_SEQ := $(CXX)
CXX_MPI := mpic++

HEADERS := csrc/include/*.h
COMMON_SOURCES :=
TARGETS := transformer-seq transformer-cube transformer-tmp transformer-tmp-cube ViT transformer-openmp

# Default build target
.PHONY: all debug release clean format check
all: release

# Rule for debug version
debug: CFLAGS += -O2 -DDEBUG
debug: $(addprefix $(OUTPUTDIR)debug-,$(TARGETS))

# Rule for release version
release: CFLAGS += -O2
release: $(addprefix $(OUTPUTDIR)release-,$(TARGETS))

# General rules for building targets with specific compilers
$(OUTPUTDIR)debug-transformer-seq $(OUTPUTDIR)release-transformer-seq: $(HEADERS) csrc/src/transformer.cpp
	@mkdir -p $(OUTPUTDIR)
	$(CXX_SEQ) -o $@ $(CFLAGS) -DSEQ $(COMMON_SOURCES) csrc/src/transformer.cpp

$(OUTPUTDIR)debug-transformer-cube $(OUTPUTDIR)release-transformer-cube: $(HEADERS) csrc/src/transformer_cube.cpp
	@mkdir -p $(OUTPUTDIR)
	$(CXX_SEQ) -o $@ $(CFLAGS) -DSEQ $(COMMON_SOURCES) csrc/src/transformer_cube.cpp

$(OUTPUTDIR)debug-transformer-tmp $(OUTPUTDIR)release-transformer-tmp: $(HEADERS) csrc/src/transformer_tmp.cpp
	@mkdir -p $(OUTPUTDIR)
	$(CXX_MPI) -o $@ $(CFLAGS) -DSEQ $(COMMON_SOURCES) csrc/src/transformer_tmp.cpp

$(OUTPUTDIR)debug-transformer-tmp-cube $(OUTPUTDIR)release-transformer-tmp-cube: $(HEADERS) csrc/src/transformer_tmp_cube.cpp
	@mkdir -p $(OUTPUTDIR)
	$(CXX_MPI) -o $@ $(CFLAGS) -DSEQ $(COMMON_SOURCES) csrc/src/transformer_tmp_cube.cpp

$(OUTPUTDIR)debug-ViT $(OUTPUTDIR)release-ViT: $(HEADERS) csrc/src/Vit.cpp
	@mkdir -p $(OUTPUTDIR)
	$(CXX_MPI) -o $@ $(CFLAGS) -DSEQ $(COMMON_SOURCES) csrc/src/Vit.cpp

$(OUTPUTDIR)debug-transformer-openmp $(OUTPUTDIR)release-transformer-openmp: $(HEADERS) csrc/src/transformer_openmp.cpp
	@mkdir -p $(OUTPUTDIR)
	$(CXX_SEQ) -o $@ $(CFLAGS) -fopenmp -DSEQ $(COMMON_SOURCES) csrc/src/transformer_openmp.cpp

format:
	clang-format -i csrc/src/*.cpp csrc/include/*.h

clean:
	rm -rf $(OUTPUTDIR)

check:	release
	./checker.py tmp -psc

FILES = csrc/src/*.cpp \
		csrc/include/*.h

handin.tar: $(FILES)
	tar cvf handin.tar $(FILES)
