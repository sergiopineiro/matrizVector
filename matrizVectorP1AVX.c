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

    int m, n, test, i, j, z, k, mpad, npad;
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
    __m256 reg_A, reg_Alfa, reg_x, reg_n1, reg_n2, reg_n3, reg_n4, reg_y;
    __m128 res1, res2;
    float arrAlfa[8] __attribute__((aligned(32))) = {alfa, alfa, alfa, alfa,
                                                    alfa, alfa, alfa, alfa};

    reg_Alfa = _mm256_load_ps(arrAlfa);

    int arrInd[8] __attribute__((aligned(32))) = {4, 5, 6, 7,
                                                0, 1, 2, 3};
    __m256i indx = _mm256_load_si256((__m256i *) arrInd);

    int w, f;
    float arr[8];

    // Parte fundamental del programa
    assert (gettimeofday (&t0, NULL) == 0);
    for (i=0; i<m; i+=8) {
        for (j=0; j<n; j+=8) {
            reg_x = _mm256_load_ps(&x[j]);
            for(z=0; z<8; z++){
                reg_A = _mm256_load_ps(&A[(i+z)*n+j]);
                reg_A = _mm256_mul_ps(reg_Alfa, reg_A);
                reg_A = _mm256_mul_ps(reg_A, reg_x);
                if(j!=0){
                    arrRegRes[i+z] = _mm256_add_ps(arrRegRes[i+z], reg_A);
                }
                else {
                    arrRegRes[i+z] = reg_A;
                }
            }
        }

        for(z=0; z<2; z++){
            for(k=0; k<2; k++){
                if(!k){
                    reg_n1 = _mm256_unpacklo_ps(arrRegRes[i+z*4], arrRegRes[i+1+z*4]);
                    reg_n2 = _mm256_unpacklo_ps(arrRegRes[i+2+z*4+2*k], arrRegRes[i+3+z*4+2*k]);
                }
                else{
                    reg_n1 = _mm256_unpackhi_ps(arrRegRes[i+z*4], arrRegRes[i+1+z*4]);
                    reg_n2 = _mm256_unpackhi_ps(arrRegRes[i+2+z*4], arrRegRes[i+3+z*4]);

                }
                
                reg_n3 = _mm256_shuffle_ps(reg_n1, reg_n2, _MM_SHUFFLE(1,0,1,0));
                reg_n4 = _mm256_shuffle_ps(reg_n1, reg_n2, _MM_SHUFFLE(3,2,3,2));

                if(!k){
                    reg_y = _mm256_add_ps(reg_n3, reg_n4);
                }
                else{
                    reg_n3 = _mm256_add_ps(reg_n3, reg_n4);
                    reg_y = _mm256_add_ps(reg_y, reg_n3);
                }
            }

            res1 = _mm256_castps256_ps128(reg_y);
            reg_y = _mm256_permutevar8x32_ps(reg_y, indx);
            res2 = _mm256_castps256_ps128(reg_y);
            res1 = _mm_add_ps(res1, res2);

            _mm_store_ps(&ypad[i+4*z], res1);
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