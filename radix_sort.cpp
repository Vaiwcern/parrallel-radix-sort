#include <stdio.h>
#include <stdint.h>
#include <cstdlib>  
#include <ctime>
#include <cstring>  

void radix_sort(int *a, int n) {
    int *bit = (int*)malloc(n * sizeof(int));
    int *dst = (int*)malloc(n * sizeof(int));
    int *nOneBefore = (int*) malloc(n * sizeof(int));

    for (int i = 0; i < 8; ++i) {
        // get bit 
        for (int j = 0; j < n; ++j) {
            bit[j] = (a[j] >> i) & 1;
        }

        nOneBefore[0] = 0;
        for(int j = 1; j < n; ++j) {
            nOneBefore[j] = nOneBefore[j - 1] + bit[j - 1];
        }

        // get new position
        int numZeros = n - nOneBefore[n - 1] - bit[n - 1];
        for (int j = 0; j < n; ++j) {
            int rank;
            if (bit[j] == 0) {
                rank = j - nOneBefore[j];
            }
            else {
                rank = numZeros + nOneBefore[j];
            }
            dst[rank] = a[j];
        }

        int* tmp = a;
        a = dst;
        dst = tmp;
    }

    free(bit);
    free(dst);
    free(nOneBefore);
}

int main() {
    srand(static_cast<unsigned int>(time(0)));

    int n = 10;
    printf("Input size: %d\n", n);

    int *a = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; ++i) {
        a[i] = rand() % 255;
    }

    printf("Before sort:\n");
    for (int i = 0; i < n; ++i) {
        printf("%d ", a[i]);
    }
    printf("\n");

    radix_sort(a, n);

    printf("After sort:\n");
    for (int i = 0; i < n; ++i) {
        printf("%d ", a[i]);
    }
    printf("\n");

    free(a);

    return 0;
}