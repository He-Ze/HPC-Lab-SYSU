g++ -g -Wall -o PthFor ParallelFor.cpp -lpthread
g++ -ggdb -Wall -shared -fpic -o libPF.so ParallelFor.cpp
g++ PFGEMM.cpp -ldl -o PFGEMM -L. -lPF -lpthread