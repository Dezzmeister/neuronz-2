package dezzy.neuronz2.russianness;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

import dezzy.neuronz2.math.constructs.Tensor3;

public class RussianDataset implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -3828943677823678052L;
	
	final Tensor3[] images;
	final double[] russiannesses;
	
	public RussianDataset(final Tensor3[] _images, final double[] _russiannesses) {
		images = _images;
		russiannesses = _russiannesses;
	}
	
	public final void saveAs(final String path) throws IOException {
		final FileOutputStream fos = new FileOutputStream(path);
		final ObjectOutputStream oos = new ObjectOutputStream(fos);
		
		oos.writeObject(this);
		oos.close();
	}
	
	public static final RussianDataset loadFrom(final String path) throws IOException, ClassNotFoundException {
		final FileInputStream fis = new FileInputStream(path);
		final ObjectInputStream ois = new ObjectInputStream(fis);
		final RussianDataset dataset = (RussianDataset) ois.readObject();
		
		ois.close();
		return dataset;
	}
}
