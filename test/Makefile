CC = mpicxx
PROJ = knndbscan
APP = $(PROJ)
CFLAGS = -O3 -I../include

.PHONY: clean

$(APP): knndbscan.cpp
	@echo ">>> Trying Intel OpenMP..."
	@$(CC) $(CFLAGS) -qopenmp knndbscan.cpp -o $(APP) 2>/dev/null || \
	( echo ">>> -qopenmp not supported. Trying GCC OpenMP..." && \
	  $(CC) $(CFLAGS) -fopenmp knndbscan.cpp -o $(APP) 2>/dev/null || \
	  ( echo "ERROR: Neither -qopenmp nor -fopenmp supported. Install OpenMP support." && false ) )

clean:
	rm -f $(APP)
