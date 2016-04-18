import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class DevelopmentSet {
	public static void main(String[] args) throws IOException{
		//read the whole training set
		List<String> trainSet = new ArrayList<String>();
		File inputFile=new File("train.csv");
		Scanner scanner = new Scanner(inputFile,"ISO-8859-1");
		String header=scanner.nextLine();
		while(scanner.hasNextLine()) {
			String line = scanner.nextLine();
			if (line.startsWith("@")||line.isEmpty()) {
				continue;
			}
			trainSet.add(line);
		}


		int trainSetSize=trainSet.size();
		int subTrainSetSize=trainSetSize*4/5;

		Random rand=new Random(System.currentTimeMillis());
		Collections.shuffle(trainSet, rand);



		// get the subTrainSet
		List<String> subTrainSet = new ArrayList<String>();
		for (int i = 0; i < subTrainSetSize; i++) {
			subTrainSet.add(trainSet.get(i));
		}
		//sort according to "id", in an ascending order
		subTrainSet.sort(new CSVStringComparator());


		// write the sub training set
		FileOutputStream fstream=new FileOutputStream("subTrainSet.csv");
		OutputStreamWriter out_ISO_8859_1=new OutputStreamWriter(fstream,"ISO-8859-1");
		BufferedWriter out=new BufferedWriter(out_ISO_8859_1);
		out.write(header);
		out.newLine();
		for (int i = 0; i < subTrainSet.size(); i++) {
			out.write(subTrainSet.get(i));
			out.newLine();
		}
		out.close();


		// get the development set
		List<String> developmentSet = new ArrayList<String>();
		for (int i = subTrainSetSize; i < trainSetSize; i++) {
			developmentSet.add(trainSet.get(i));
		}
		//sort according to "id", in an ascending order
		developmentSet.sort(new CSVStringComparator());


		// write the development set
		fstream=new FileOutputStream("developmentTestSet.csv");
		out_ISO_8859_1=new OutputStreamWriter(fstream,"ISO-8859-1");
		out=new BufferedWriter(out_ISO_8859_1);
		out.write(header);
		out.newLine();
		for (int i = 0; i < developmentSet.size(); i++) {

			out.write(developmentSet.get(i));
			out.newLine();
		}
		out.close();



	}


}

// compare according to the first column "id", in an ascending order
class CSVStringComparator implements Comparator<String>{

	public int compare(String a, String b) {
		// a.newPoint and b.newPoint must be the same point
		String[]array_A=a.split(",");
		String[]array_B=b.split(",");
		// compare the "id"
		double id_A=Double.parseDouble(array_A[0]);
		double id_B=Double.parseDouble(array_B[0]);

		// sort in an ascending order
		return id_A < id_B ? -1 : (id_A == id_B ? 0 : 1);
	}
}
