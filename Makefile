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

COMMON_SOURCES := csrc/src/dropout.cpp \
	csrc/src/layernorm.cpp \
	csrc/src/feedforward.cpp
HEADERS := csrc/include/*.h

TARGETBIN := transformer-$(CONFIGURATION)-seq transformer-$(CONFIGURATION)-tmp

.SUFFIXES:
.PHONY: all clean

all: $(TARGETBIN)

transformer-$(CONFIGURATION)-seq: $(HEADERS) csrc/src/transformer.cpp
	$(CXX) -o $@ $(CFLAGS) -DSEQ $(COMMON_SOURCES) csrc/src/transformer.cpp

transformer-$(CONFIGURATION)-tmp: $(HEADERS) $(COMMON_SOURCES) csrc/src/transformer_tmp.cpp
	mpic++ -o $@ $(CFLAGS) -DSEQ $(COMMON_SOURCES) csrc/src/transformer_tmp.cpp

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
