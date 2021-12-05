default:
	@echo "Usage:"
	@echo "   p1c:     Practica 1 sin optimizaciones"
	@echo "   autovec    Practica 1 con vectorizacion automatica"
	@echo "   p1AVX:   Practica 1 con vectorizacion manual con AVX"
	@echo "   p1SSE:   Practica 1 con vectorizacion manual con SSE"
all: pSSEc autovec manvec1 manvec2 manvec3

p1c: matrizVectorP1.c
	gcc -O3 -fno-tree-vectorize -o matrizVectorP1 matrizVectorP1.c -lm

autovec: matrizVectorP1.c
	gcc -O3 -march=native -o matrizVectorP1Vec matrizVectorP1.c -lm

p1AVX: matrizVectorP1AVX.c
	gcc  -O3 -march=native -mavx2 -o matrizVectorP1AVX matrizVectorP1AVX.c -lm

p1SSE: matrizVectorP1SSE.c
	gcc -O3 -march=native -msse -o matrizVectorP1SSE matrizVectorP1SSE.c -lm
clean:
	-rm p3c matrizVectorP1 matrizVectorP1Vec matrizVectorP1SSE matrizVectorP1AVX

