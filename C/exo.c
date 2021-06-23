#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*exo2*/
double moyenmatrice(double resultatnote[20][50], int l,int c)
{
  double moyenne;
   double somme=0;
    for (int i=0;i<l;i++)
    {
    for (int j=0;j<c;j++)
    {
      somme= somme+ resultatnote[i][j];
    }
    }
 moyenne= somme/(l*c);
 return moyenne;

 }
 double SommeDiagonalMatrice(double resultatnote[20][50], int l,int c){
   double somme=0;
    for (int i=0;i<l;i++)
    {
      somme = somme+ resultatnote[i][i];
    }
 return somme;
 }

 void affiche(double resultatnote[20][50], int l, int c){
    for(int i=0;i<l; i++)
    {
        for(int j=0;j<c; j++)
        {
           printf("%f ",resultatnote[i][j]);
        }

    }
    printf("\n");
}

 int main(){
    int l = 20;
    int c = 50;
    double moy;
    double resultatnote[20][50];
    srand(time (NULL));
    
    for (int i=0; i<l;i++)
    {
    for (int j=0; j<c;j++)
    {
     resultatnote[i][j]= (rand()/(double)RAND_MAX)*(20);
    }
    }
    //affiche(resultatnote,l,c);
    moy= moyenmatrice(resultatnote, l, c);
    printf ("la moyenne est: %f \n",moy );
    printf("La somme de diagonal est: %f\n", SommeDiagonalMatrice(resultatnote, l, c));

 }
