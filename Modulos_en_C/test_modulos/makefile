.PHONY: default help objects executable all clean
CC = gcc

CC_FLAGS = -g -std=gnu99 -O3 -Wall
LD_FLAGS = -lm 

LD = $(CC)

SOURCE_C = $(wildcard ../*.c)
OBJECTS_C = $(patsubst ../%.c, %_c.o, $(SOURCE_C))
EXECUTABLE = test_modulos.e

default: all
objects: $(OBJECTS_C)
executable: $(EXECUTABLE)

all: objects executable

%_c.o: ../%.c
	$(CC) $(CC_FLAGS) -c $^ -o $@

$(EXECUTABLE): $(OBJECTS_C)
	$(LD) $^ -o $@ $(LD_FLAGS)

help:
	@echo "\
Options:\n\n\
  make objects:       compiler makes objects for every *.c\n\
  make executable:    compiler makes executable\n\
  make all:           build all previous\n\
  make clean:         delete output files\n\
  make help:          display this help"


clean:
	rm $(OBJECTS_C) $(EXECUTABLE)
