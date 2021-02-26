import java.util.*; 
public class Factor {
  private static String operation = "2 . 5 + 3 ( 2 ( 2 - 5 ) + 1 ) + 4 end";
  public static void main(String[] arg){
    Factor factor = new Factor();

    System.out.println("Helloooo "+factor.factoriel(50));
    
  }
  public double function(double x){
      //return 4*Math.pow(x,4)+3*Math.pow(x,3)+4*Math.pow(x,2)+4*x+6;
      //return  Math.log(2*x+3)-3*x-1;
      //return Math.pow(x,3)-3*Math.pow(x,2)+1;

      //return (x-3.5567)*(x+1.345)*(x-7.819)*(x+2)*x;
      //return Math.pow(x-1,2)*(x+2)*Math.pow(x+1,3);
      return Math.pow(x,6)+3*Math.pow(x,5)-6*Math.pow(x,3)-3*Math.pow(x,2)+3*x+2;//(x-1)^2(x+2)(x+1)^3
  }
  public long factorieln(long n){
    if(n == 1 || n == 0) return 1;
    else return n * factoriel(n-1);
  }
  public static String factoriel(int n) {
    double d = 0;
    for (int i = 2; i <= n; i++ )
        d += Math.log10(i);
    int p = (int) Math.floor(d);
    return String.format(Locale.US, "%.16fe+%s%d", Math.pow(10, d - p), (p < 10 ? "0" : ""), p);
}
}