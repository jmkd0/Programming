
import java.util.*; 
import java.util.Arrays;
public class Function {
  public boolean error = false;
  private static int index;
  private String[] operation;
  private List<String> digitString;

  public Function(String[] operation ){
    this.index = 0;
    this.operation = operation;
    this.digitString = new ArrayList<String>(Arrays.asList("0","1","2","3","4","5","6","7","8","9","."));
  }
  public double evaluation(){
    //String equation = "2 ( 2 + 1 ) end";
    String equation = ") ( 2 + 3 √ ( 4 + 3 ) + ( 1 ) + 2 ^ 2 ! end";
    this.operation = equation.split(" ");
    int openPar = 0;
    int closePar = 0;

    for(int i=0; i< this.operation.length; i++){
      if(this.operation[i].equals("(")) openPar++;
      if(this.operation[i].equals(")")) closePar++;
    }
    if(openPar < closePar){
      this.error = true;
      return 0;
    }
      return this.sommeSubstring();
  }
  public double sommeSubstring(){
      double value = this.productDivision();
      while(true){
          if(this.operation[index-1].equals("+"))
            value += this.productDivision();
          else if(this.operation[index-1].equals("-"))
            value -= this.productDivision();
          else return value;
      }
  }
  public double productDivision(){
      double value = this.getScalarValue();
      while(true){
          if(this.operation[this.index-1].equals("*"))
            value *= this.getScalarValue();
          else if (this.operation[this.index-1].equals("(")){
            this.index--;
            value *= this.getScalarValue();
          }
          else if(this.operation[this.index-1].equals("/"))
            value /= this.getScalarValue();
          else return value;
      }
  }
  public double getScalarValue(){
      boolean negate = false;
      String token;
      double value = 0;
      String str= new String();
      token = this.operation[index++];
      
      if(token.equals("(")){
          value = this.sommeSubstring();
          if(this.operation[index-1].equals(")")) index++;
      }else{
        token = this.operation[index-1];
        if(token.equals("+") || token.equals("-")){
          negate = token.equals("-");
          this.index++;
          System.out.println("token");
        } 
        token = this.operation[index-1];
        if(this.digitString.contains(token)){
            while (this.digitString.contains(token)) {
                str += token;
                token = this.operation[index++];
            }
            if(str.length()-str.replace(".","").length()>1) {this.error = true; return 0;}
            value = Double.parseDouble(str.toString());
            System.out.println(value);
        }
        
      }
      //if(this.operation[index-1].equals(")")) index++;
      if(token.equals("π")){
        int cpt = 1;
        token = this.operation[this.index++];
        if(token.equals("π")){
          while(token.equals("π")){ 
            cpt++;
            token = this.operation[this.index++];
          }
        }else if(this.digitString.contains(token)){this.error = true; return 0;}
        value = Math.pow(Math.PI, cpt);
      }
      if(token.equals("²")){
         value = Math.pow(value, 2);
         token = this.operation[this.index++];
         if(this.digitString.contains(token)){this.error = true; return 0;}
      }
      if(token.equals("³")){
         value = Math.pow(value, 3);
         token = this.operation[this.index++];
         if(this.digitString.contains(token)){this.error = true; return 0;}
      }
      if(token.equals("√")){
        String noAtLeft = "-+*/√(";
        String noAtRight = "!*/²³)";
        String l = this.operation[index-2];
        String r = this.operation[index];
        if(noAtLeft.length()-noAtLeft.replace(l,"").length()==0 && 
          noAtRight.length()-noAtRight.replace(r,"").length()==0 ){
            value *= Math.sqrt(this.getScalarValue());
          }else {this.error = true; return 0;}
      }
      if(token.equals("^")){
        String noAtLeft = "-+*/√(";
        String noAtRight = "!+-*/²³)";
        String l = this.operation[index-2];
        String r = this.operation[index];
        if(noAtLeft.length()-noAtLeft.replace(l,"").length()==0 && 
          noAtRight.length()-noAtRight.replace(r,"").length()==0 ){
            value = Math.pow(value, this.getScalarValue());
          }else {this.error = true; return 0;}
      }
      if(token.equals("!")){
        String noAtLeft = "-+*/√(";
        String noAtRight = "π²³";
        String l = this.operation[index-2];
        String r = this.operation[index];
        if(noAtLeft.length()-noAtLeft.replace(l,"").length()==0 && 
          noAtRight.length()-noAtRight.replace(r,"").length()==0 && 
          !this.digitString.contains(r) && value == (int)value &&
          value >=0 && value <= 100){
            value = this.factorial(value);
            this.index++;
          }else {this.error = true; return 0;}
      }
      if(token.equals("ln")){
        String noAtLeft = "";
        String noAtRight = "!)²³+-*/^";
        String l = this.operation[index-2];
        String r = this.operation[index];
        if(noAtLeft.length()-noAtLeft.replace(l,"").length()==0 && 
          noAtRight.length()-noAtRight.replace(r,"").length()==0 && 
          value > 0){
            value *= Math.log(value);
            this.index++;
          }else {this.error = true; return 0;}
      }
      if(token.equals("exp")){
        String noAtLeft = "-+*/√(";
        String noAtRight = "π²³";
        String l = this.operation[index-2];
        String r = this.operation[index];
        if(noAtLeft.length()-noAtLeft.replace(l,"").length()==0 && 
          noAtRight.length()-noAtRight.replace(r,"").length()==0 && 
          !this.digitString.contains(r) && value == (int)value &&
          value >=0 && value <= 100){
            value *= Math.exp(this.getScalarValue());
            this.index++;
          }else {this.error = true; return 0;}
      }
      /*switch (token){()
        case "exp":
            index++;
            break;
        case "sin":
            index++;
            break;
        case "cos":
            index++;
            break;
        case "tan":
            index++;
            break;
        case "asin":
            index++;
            break;
        case "acos":
            index++;
            break;
        case "atan":
            index++;
            break;
          default:
      }*/
      //System.out.println(index);
      if(negate) value = -value;
      return value;
  }
  public double factorial(double d){
    if(d == 0 || d == 1) return 1;
    else return d * factorial(d-1);
  }
}