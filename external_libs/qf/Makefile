AR       ?= ar
CXX      ?= g++
CXXFLAGS  = -O2 -DNDEBUG -fPIC

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S)-$(UNAME_M),Darwin-x86_64)
 CXXFLAGS += -arch x86_64
endif

qf.a: qfc.o
	${AR} rcs $@ $^ 

%.o: %.cpp
	${CXX} ${CXXFLAGS} -o $@ -c $< ${INC}

clean:
	rm -f *.o *.a
