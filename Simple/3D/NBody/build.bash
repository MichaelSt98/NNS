set -x
opt="-O3 -march=native -mtune=native"
#opt="-O2"
#pg="-pg"
#g="-g"
g++ $pg $g -std=c++17 -Wall -Wextra $opt BarnzNhutt.cpp -c -o run.o -Xpreprocessor -fopenmp
g++ $pg $g run.o -o run -Xpreprocessor -fopenmp
