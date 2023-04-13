OUTPUTDIR := bin/

# CFLAGS := -std=c++14 -fvisibility=hidden -lpthread -g -p -fno-omit-frame-pointer
CFLAGS := -std=c++14 -fvisibility=hidden -lpthread

ifeq (,$(CONFIGURATION))
	CONFIGURATION := release
endif

ifeq (debug,$(CONFIGURATION))
CFLAGS += -g
else
CFLAGS += -O2
endif

SOURCES := csrc/src/*.cpp
HEADERS := csrc/include/*.h

TARGETBIN := transformer-$(CONFIGURATION)

.SUFFIXES:
.PHONY: all clean

all: $(TARGETBIN)

$(TARGETBIN): $(SOURCES) $(HEADERS)
	$(CXX) -o $@ $(CFLAGS) $(SOURCES) 

format:
	clang-format -i csrc/src/*.cpp csrc/include/*.h

clean:
	rm -rf ./transformer-$(CONFIGURATION)

check:	default
	./checker.pl

FILES = csrc/src/*.cpp \
		csrc/include/*.h

handin.tar: $(FILES)
	tar cvf handin.tar $(FILES)
