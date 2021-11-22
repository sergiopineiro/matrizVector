#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

int main( int argc, char *argv[] ) {

    int m, n, test, i, j;
    float alfa;
    struct timeval t0, t1, t;

    // Parámetro 1 -> m
    // Parámetro 2 -> n
    // Parámetro 3 -> alfa
    // Parámetro 4 -> booleano que nos indica si se desea imprimir matrices y vectores de entrada y salida
    if(argc>3){
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        alfa = atof(argv[3]);
        test = atoi(argv[4]);
    }
    else{
        printf("NUMERO DE PARAMETROS INCORRECTO\n");
	printf("Uso: ./matrizVectorP1 <n> <m> <alfa> <debug>\n");
	printf("\t<m>\t-> número de filas de la matriz\n");
	printf("\t<n>\t-> número de columnas de la matriz / elementos del vector\n");
	printf("\t<alfa>\t-> parámetro de escalado\n" );
	printf("\t<debug>\t-> indica si deben imprimirse las matrices y vectores de entrada y salida (0/1)\n");
        exit(0);
    }

    float *x = (float *)malloc(n*sizeof(float));
    float *A = (float *)malloc(m*n*sizeof(float));
    float *y = (float *)malloc(m*sizeof(float));

    // Se inicializan la matriz y los vectores

    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            A[i*n+j] = ((float)(1+i+j))/m/n;
        }
    }

    for(i=0; i<n; i++){
        x[i] = ((float)(1+i))/n;
    }

    for(i=0; i<m; i++){
        y[i] = ((float)(1-i))/m;
    }

    if(test){
        printf("\nMatriz A es...\n");
        for(i=0; i<m; i++){
            for(j=0; j<n; j++){
                printf("%f ", A[i*n+j]);
            }
            printf("\n");
        }

        printf("\nVector x es...\n");
        for(i=0; i<n; i++){
            printf("%f ", x[i]);
        }
        printf("\n");

        printf("\nVector y al principio es...\n");
        for(i=0; i<m; i++){
            printf("%f ", y[i]);
        }
        printf("\n");
    }

    // Parte fundamental del programa
    assert (gettimeofday (&t0, NULL) == 0);
    for (i=0; i<m; i++) {
        for (j=0; j<n; j++) {
            y[i] += alfa*A[i*n+j]*x[j];
        }
    }
	
    assert (gettimeofday (&t1, NULL) == 0);
    timersub(&t1, &t0, &t);

    if(test){
        printf("\nAl final vector y es...\n");
        for(i=0; i<m; i++){
            printf("%f ", y[i]);
        }
        printf("\n");

        float *testy = (float *) malloc(m*sizeof(float));
        for(i=0; i<m; i++){
            testy[i] = ((float)(1-i))/m;
        }

        // Se calcula el producto sin ninguna vectorización
        for (i=0; i<m; i++) {
            for (j=0; j<n; j++) {
                testy[i] += alfa*A[i*n+j]*x[j];
            }
        }

        int errores = 0;
        for(i=0; i<m; i++){
	    if( (testy[i]-y[i])*(testy[i]-y[i]) > 1e-10 ) {
                errores++;
                printf("\n Error en la posicion %d porque %f != %f", i, y[i], testy[i]);
            }
        }
        printf("\n%d errores en el producto matriz vector con dimensiones %dx%d\n", errores, m, n);
        free(testy);
    }

    printf ("Tiempo      = %ld:%ld(seg:mseg)\n", t.tv_sec, t.tv_usec/1000);

    free(x);
    free(y);
    free(A);
	
    return 0;
}




