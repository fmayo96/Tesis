.PHONY: default help objects sharedobjects all clean
CC = gcc
CC_FLAGS = -fPIC -O3 
LD_FLAGS = -shared 

LD = $(CC)

SOURCE_C = $(wildcard *.c)
OBJECTS_C = $(patsubst %.c, %_c.o, $(SOURCE_C))
SHARED_OBJECTS = libtime_evolution.so

default: all
objects: $(OBJECTS_C)
sharedobjects: $(SHARED_OBJECTS)
all: objects sharedobjects

%_c.o: %.c
	$(CC) $(CC_FLAGS) -c $^ -o $@

$(SHARED_OBJECTS): $(OBJECTS_C)
	$(LD) $(LD_FLAGS) $^ -o $@
	nm -n libtime_evolution.so
help:
	@echo "\
Options:\n\n\
  make objects:       compiler makes objects for every *.c\n\
  make sharedobjects:    compiler makes sharedobjects\n\
  make all:           build all previous\n\
  make clean:         delete output files\n\
  make help:          display this help"


clean:
	rm $(OBJECTS_C) $(SHARED_OBJECTS)
