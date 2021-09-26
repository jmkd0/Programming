#include <stdio.h>
#include <stdlib.h>

void draw_triangle(int N){
    for(int i=0; i<N; i++){
        for(int j=0; j<N*2; j++){
            if(j==N-i || j==N+i){
                printf("*");
            }else{
                printf(" ");
            }
        }
        printf("\n");
    }
}
int main(){
    int N=1;
    draw_triangle(N);
    return 0;
}
