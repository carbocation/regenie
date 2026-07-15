FLAG=-O2 -DNDEBUG -fPIC -std=legacy
AR   ?= ar
ifeq ($(origin FC), default)
 FC := gfortran
endif

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S)-$(UNAME_M),Darwin-x86_64)
 FLAG += -arch x86_64
endif

OBJECTS = $(patsubst %.f,%.o,$(wildcard ./*.f))

libquad.a: ${OBJECTS}
	${AR} rcs $@ $^ 

%.o: %.f
	${FC} $(FLAG) -c $<

clean:
	rm -f *.o *.a
