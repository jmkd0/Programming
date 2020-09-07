import java.util.*; 
public class Main {
  private static String operation = "2 . 5 + 3 ( 2 ( 2 - 5 ) + 1 ) + 4 end";
  public static void main(String[] arg){
    String[] listOperation = operation.split(" ");
    Function function = new Function(listOperation);
    double value = function.evaluation();
    if(!function.error){
      System.out.println("Le calcule d'opération donne: "+value);
    }else{
      System.out.println("Error");
    }
    
    
  }
}
/* 
 */