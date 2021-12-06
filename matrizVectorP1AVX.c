#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <immintrin.h>


int main( int argc, char *argv[] ) {

    int m, n, test, i, j, z, mpad, npad;
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

    mpad = m%8==0? m : m + (8-m%8);
    npad = n%8==0? n: n + (8-n%8);
    float *x = (float *)_mm_malloc(npad*sizeof(float), 32);
    float *A = (float *)_mm_malloc(mpad*npad*sizeof(float), 32);
    float *ypad = (float *)_mm_malloc(mpad*sizeof(float), 32);
    float *y = (float *)malloc(m*sizeof(float));

    // Se inicializan la matriz y los vectores

    for(i=0; i<mpad; i++){
        for(j=0; j<npad; j++){
            if(i<m & j<n){
                A[i*n+j] = ((float)(1+i+j))/m/n;
            }
            else{
                A[i*npad+j] = 0.0;
            }
        }
    }

    for(i=0; i<npad; i++){
        if(i<n){
            x[i] = ((float)(1+i))/n;
        }
        else{
            x[i] = 0.0;
        }
        
    }

    for(i=0; i<m; i++){
        y[i] = ((float)(1-i))/m;
    }

    for (i=0; i<mpad; i++){
        ypad[i] = 0.0;
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
    
    float arrIni[8] __attribute__((aligned(32))) = {0.0, 0.0, 0.0, 0.0,
                                                    0.0, 0.0, 0.0, 0.0};
    __m256 arrRegRes[mpad];
    //Inicializo array de registros
    for(i=0; i<mpad; i++){
        arrRegRes[i] = _mm256_load_ps(arrIni);
    }
    __m256 reg_A, reg_Alfa, reg_x, reg_y;
    float arrAlfa[8] __attribute__((aligned(32))) = {alfa, alfa, alfa, alfa,
                                                    alfa, alfa, alfa, alfa};

    reg_Alfa = _mm256_load_ps(arrAlfa);

    // Parte fundamental del programa
    assert (gettimeofday (&t0, NULL) == 0);
    for (i=0; i<m; i++) {
        for (j=0; j<n; j+=8) {
            reg_A = _mm256_load_ps(&A[i*n+j]);
            reg_A = _mm256_mul_ps(reg_Alfa, reg_A);
            reg_x = _mm256_load_ps(&x[j]);
            reg_A = _mm256_mul_ps(reg_A, reg_x);
            if(j!=0){
                reg_y = _mm256_add_ps(reg_y, reg_A);
            }
            else {
                reg_y = reg_A;
            }
        }
        arrRegRes[i]= reg_y;
    }

    int arrInd[8] __attribute__((aligned(32))) = {0, 1, 4, 5,
                                                2, 3, 6, 7};
    __m256i indx = _mm256_load_si256((__m256i *) arrInd);

    for(i=0; i<mpad; i+=8){
        __m256 res1, res2;
        __m128 res_128_1, res_128_2;

        for(j=0; j<2; j++){
            res1 = _mm256_hadd_ps(arrRegRes[i+4*j], arrRegRes[i+1+4*j]);
            res2 = _mm256_hadd_ps(arrRegRes[i+2+4*j], arrRegRes[i+3+4*j]);
         
            res1 = _mm256_permutevar8x32_ps(res1, indx);
            res2 = _mm256_permutevar8x32_ps(res2, indx);
            
            res1 = _mm256_hadd_ps(res1, res1);
            res1 = _mm256_hadd_ps(res1, res1);
            res1 = _mm256_permutevar8x32_ps(res1, indx);
            
            res2 = _mm256_hadd_ps(res2, res2);
            res2 = _mm256_hadd_ps(res2, res2);
            res2 = _mm256_permutevar8x32_ps(res2, indx);

            res_128_1 = _mm256_castps256_ps128(res1);
            res_128_2 = _mm256_castps256_ps128(res2);
            res_128_1 = _mm_shuffle_ps(res_128_1, res_128_2, _MM_SHUFFLE(2,0,2,0));

            _mm_store_ps(&ypad[i+4*j], res_128_1);
        }
    }

    for (i=0; i<m; i++){
        y[i] += ypad[i];
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

    _mm_free(A);
    _mm_free(x);
    _mm_free(ypad);
    free(y);
	
    return 0;
}