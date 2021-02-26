//import javafx.util.Pair;
import java.util.List;
 
import java.util.Vector; 
import java.util.Collections;
import java.util.Arrays;
import java.util.ArrayList;
public class Equation {
  // Définition de la fonction main
  public static void main(String[] arg){
    //List<String> listOf4Names = Arrays.asList("A1","A2","A3","A4");
    //ArrayList  lis = new ArrayList();
    ArrayList  lis = new ArrayList(Arrays.asList(4,7));
    //ArrayList<Integer>  lis = new ArrayList<Integer>(Arrays.asList(4,7));
    
    lis.add("voila");
    lis.set(1, 45);

    for(int i=0; i<lis.size(); i++){
      System.out.println(lis.get(i));
    }
   /*  for(int a: val){
    System.out.println(a);
     } */

    /* Vector<Integer> vec = new Vector<Integer>(Arrays.asList(val));
    //Vector<Integer> vec = new Vector<Integer>();
    vec.add(67); 

    Collections.sort(vec);
    vec.set(3, 45);
    for(int i=0; i<vec.size(); i++){
      System.out.println(vec.get(i));
    }
   

  
System.out.println(vec.size()); */


    //Equation equation = new Equation();
    //equation.resolution();
    
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
      //return 4*Math.pow(x,4)+3*Math.pow(x,3)+4*Math.pow(x,2)+4*x+6;
      return  Math.log(2*x+3)-3*x-1;
      //return Math.pow(x,3)-3*Math.pow(x,2)+1;
      //return (x-3.5567)*(x+1.345)*(x-7.819)*(x+2)*x;
  }
}