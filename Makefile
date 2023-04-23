OUTPUTDIR := bin/

# CFLAGS := -std=c++14 -fvisibility=hidden -lpthread -g -p -fno-omit-frame-pointer
CFLAGS := -std=c++14 -fvisibility=hidden -lpthread -Wall -Wextra

ifeq (,$(CONFIGURATION))
	CONFIGURATION := release
endif

ifeq (debug,$(CONFIGURATION))
CFLAGS += -g
else
CFLAGS += -O2
endif

COMMON_SOURCES :=
HEADERS := csrc/include/*.h

TARGETBIN := transformer-$(CONFIGURATION)-seq transformer-$(CONFIGURATION)-tmp transformer-$(CONFIGURATION)-cube transformer-$(CONFIGURATION)-tmp-cube

.SUFFIXES:
.PHONY: all clean

all: $(TARGETBIN)

transformer-$(CONFIGURATION)-seq: $(HEADERS) csrc/src/transformer.cpp
	$(CXX) -o $@ $(CFLAGS) -DSEQ $(COMMON_SOURCES) csrc/src/transformer.cpp

transformer-$(CONFIGURATION)-tmp: $(HEADERS) $(COMMON_SOURCES) csrc/src/transformer_tmp.cpp
	mpic++ -o $@ $(CFLAGS) -DSEQ $(COMMON_SOURCES) csrc/src/transformer_tmp.cpp

transformer-$(CONFIGURATION)-cube: $(HEADERS) $(COMMON_SOURCES) csrc/src/transformer_cube.cpp
	$(CXX) -o $@ $(CFLAGS) -DSEQ $(COMMON_SOURCES) csrc/src/transformer_cube.cpp

transformer-$(CONFIGURATION)-tmp-cube: $(HEADERS) $(COMMON_SOURCES) csrc/src/transformer_tmp_cube.cpp
	mpic++ -o $@ $(CFLAGS) -DSEQ $(COMMON_SOURCES) csrc/src/transformer_tmp_cube.cpp

format:
	clang-format -i csrc/src/*.cpp csrc/include/*.h

clean:
	rm -rf ./transformer-$(CONFIGURATION)*

check:	default
	./checker.py

FILES = csrc/src/*.cpp \
		csrc/include/*.h

handin.tar: $(FILES)
	tar cvf handin.tar $(FILES)
