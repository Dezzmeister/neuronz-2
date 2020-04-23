package dezzy.neuronz2.dataio;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.zip.DataFormatException;

import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.constructs.Tensor3;

/**
 * Loads the MNIST handwritten digit dataset as an array of rank 3 tensors instead of an array of vectors
 * (like {@link MnistLoader}). Because the images are grayscale, each tensor has one matrix (color channel).
 * Loading the MNIST data as tensors allows it to be easily used with convolutional neural nets.
 *
 * @author Joe Desmond
 */
public class MnistConvLoader {
	
	/**
	 * MNIST image file identifier
	 */
	private static final int IMAGE_MAGIC_NUMBER = 0x803;
	
	/**
	 * Loads the MNIST data from a file and returns a {@link Tensor3} array, where each tensor contains
	 * an image. Labels for the data can be loaded with {@link MnistLoader#loadLabels(String)}.
	 * 
	 * @param imagePath path to MNIST image file
	 * @return MNIST images
	 * @throws IOException if there is a problem reading the file
	 * @throws DataFormatException if there is a problem interpreting the data
	 */
	public static final Tensor3[] loadImages(final String imagePath) throws IOException, DataFormatException {		
		byte[] file = Files.readAllBytes(Paths.get(imagePath));
		
		int magicNum = MnistLoader.concatenateUnsignedBytes(file[0], file[1], file[2], file[3]);
		
		if (magicNum != IMAGE_MAGIC_NUMBER) {
			throw new DataFormatException("MNIST Image magic number is not equal to expected magic number!");
		}
		
		final int items = MnistLoader.concatenateUnsignedBytes(file[4], file[5], file[6], file[7]);
		final int rows = MnistLoader.concatenateUnsignedBytes(file[8], file[9], file[10], file[11]);
		final int cols = MnistLoader.concatenateUnsignedBytes(file[12], file[13], file[14], file[15]);
		
		final int offset = 16;
		
		final Tensor3[] data = new Tensor3[items];
		
		for (int i = 0; i < items; i++) {
			final double[][] image = new double[rows][cols];
			
			for (int row = 0; row < rows; row++) {
				for (int col = 0; col < cols; col++) {
					int pixel = (row * cols) + col;
					int pixValue = MnistLoader.unsigned(file[(pixel + offset + (i * rows * cols))]);
					image[row][col] = (pixValue / (double)255.0) - 0.5;
				}
			}
			
			data[i] = new Tensor3(new Matrix(image));
		}
		
		return data;
	}
}
