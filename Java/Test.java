import java.util.*; 
import java.util.Date;
import java.text.SimpleDateFormat;
public class Test {
  public static void main(String[] arg){
    int number = 930101;//335313;//310106
    String birthday = String.valueOf(number).substring(4,6) + "/"+((Integer.parseInt(String.valueOf(number).substring(2,4)) < 50) ? String.valueOf(number).substring(2,4) : String.valueOf(Integer.parseInt(String.valueOf(number).substring(2,4))-50)) + "/19"+ String.valueOf(number).substring(0,2);
    String gender = (Integer.parseInt(String.valueOf(number).substring(2,4)) < 50) ? "Male" : "Female";
    //Date date = new SimpleDateFormat("dd/MM/yyyy").parse(birthday);
    String trans_date = String.valueOf(number).substring(4,6) + "/"+String.valueOf(number).substring(2,4)+ "/19"+ String.valueOf(number).substring(0,2);
    System.out.println(trans_date);
  }
}
//formatter.parse( )

//new SimpleDateFormat("dd/MM/yyyy").parse(String.valueOf(row1.birth_number).substring(4,6) + "/"+((Integer.parseInt(String.valueOf(row1.birth_number).substring(2,4)) < 50) ? String.valueOf(row1.birth_number).substring(2,4) : String.valueOf(Integer.parseInt(String.valueOf(row1.birth_number).substring(2,4))-50)) + "/19"+ String.valueOf(row1.birth_number).substring(0,2))
//new SimpleDateFormat("dd/MM/yyyy").parse(String.valueOf(row3.date).substring(4,6) + "/"+String.valueOf(row3.date).substring(2,4)+ "/19"+ String.valueOf(row3.date).substring(0,2))