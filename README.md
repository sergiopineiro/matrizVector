# matrizVector

En este repositorio se expone la realización del producto matriz-vector mediante extensiones multimedia SIMD, más concretamente con las instrucciones SSE y AVX. Para todas las implementaciones se permite cualquier número de matriz y vector. 
En el repositorio también se encuentra el [informe](Matriz-Vector-SIMD-Informe.pdf) sobre la comparación de las diferentes evoluciones que se realizaron sobre el código.

Para cada tipo de instrucciones, SSE y AVX, existen tres versiones: una donde sólo se vectoriza el lazo interno mediante hadds, una segunda donde se vectorizan los dos lazo  con hadds y otra final donde se vectorizan los dos lazos sin utilizar hadds. Estas versiones se pueden consultar con los siguientes tags:

- SSEv0: Intrucciones SSE, solo lazo interno vectorizado utilizando hadds.
- SSEv1: Intrucciones SSE, dos lazos vectorizados utilizando hadds.
- SSEv2: Intrucciones SSE, dos lazos vectorizados sin utilizar hadds.
- AVXv0: Intrucciones AVX, solo lazo interno vectorizado utilizando hadds.
- AVXv1: Intrucciones AVX, dos lazos vectorizado utilizando hadds.
- AVXv2: Intrucciones AVX, dos lazos vectorizado sin utilizar hadds.