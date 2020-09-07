public class Equation {
  // Définition de la fonction main
  public static void main(String[] arg){
    Equation equation = new Equation();
    equation.resolution();
    
  }


  public void resolution(){
      double start = -100;
      double end = 100;
      double step = 0.01;
      for(double d = start; d < end; d += step){
          double a = d;
          double b = d + step;
          if(function(a)*function(b) <= 0){
              System.out.println("coucou j'ai trouvé: "+dychotomy(a, b)+"\n");
          }
      }
      //System.out.println("Hello world\nd= "+function(0.4));
  }
  public double dychotomy(double a, double b){
      double precision = 0.00001;
      while(b-a > precision){
          double middle = (b+a)/2;
          if(function(a)*function(middle) <= 0) 
                b = middle;
           else a = middle;
      }
    return a;
  }



  public double function(double x){
      return  Math.log(2*x+3)-3*x-1;
      //return Math.pow(x,3)-3*Math.pow(x,2)+1;
      //return (x-3.5567)*(x+1.345)*(x-7.819)*(x+2)*x;
  }
}