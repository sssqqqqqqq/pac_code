
CXX=dpcpp

gpp_obj = main.o

CXXFLAGS = -Ofast -qopenmp -I/opt/intel/oneapi/mpi/latest/include 
LDFLAGS = -L/opt/intel/oneapi/mpi/latest/lib -lmpi -lmpicxx 

all: main
EXEC= main


main: $(gpp_obj)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(gpp_obj) -o $(EXEC).exe

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@


clean:
	rm -f *.o $(EXEC).exe
