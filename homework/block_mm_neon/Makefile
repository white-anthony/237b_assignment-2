CC      = gcc
CFLAGS  = -g -Wall -O3 -mfpu=neon -mfloat-abi=hard
INCFLAGS := -I../../helper_lib
LDFLAGS  := ../../helper_lib/helper_lib.a -lm

ifeq ($(shell uname -o), Darwin)
	LDFLAGS += -framework OpenCL
else ifeq ($(shell uname -o), GNU/Linux)
	LDFLAGS  += -L/usr/local/cuda/lib64 -lOpenCL
	INCFLAGS += -I/usr/local/cuda/include
else
	LDFLAGS += -lOpenCL
endif

all: solution

solution: ../../helper_lib/helper_lib.a main.c
	$(CC) $(CFLAGS) -o $@ $^ $(INCFLAGS) $(LDFLAGS)

../../helper_lib/helper_lib.a:
	cd ../../helper_lib && $(MAKE) 

run: solution
	@for i in $(shell seq 0 9); do \
		echo "Running test $${i}"; \
		./solution ../Dataset_a/$${i}/input0.raw ../Dataset_a/$${i}/input1.raw ../Dataset_a/$${i}/output.raw output.raw || exit 1; \
	done

clean: 
	rm -f solution
	cd ../../helper_lib && $(MAKE) clean
