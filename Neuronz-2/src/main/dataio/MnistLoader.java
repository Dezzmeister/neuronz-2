package main.dataio;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import main.math.constructs.Vector;

/**
 * Loads MNIST handwritten image data.
 *
 * @author Joe Desmond
 */
public final class MnistLoader {
	
	/**
	 * MNIST image file identifier
	 */
	private static final int IMAGE_MAGIC_NUMBER = 0x803;
	
	/**
	 * MNIST label file identifier
	 */
	private static final int LABEL_MAGIC_NUMBER = 0x801;
	
	/**
	 * Loads MNIST images from a file and returns a Vector array containing each image. Each image is stored in one Vector with pixels corresponding to Vector components.
	 * The pixel colors are grayscale, normalized doubles.
	 * 
	 * @param imagePath path to image file
	 * @return MNIST images
	 */
	public final static Vector[] loadImages(final String imagePath) {
		Vector[] data = null;
		
		try {
			byte[] file = Files.readAllBytes(Paths.get(imagePath));
			int magicNum = concatenateUnsignedBytes(file[0], file[1], file[2], file[3]);
			
			if (magicNum != IMAGE_MAGIC_NUMBER) {
				System.out.println("MNIST Image magic number is not equal to expected magic number!");
			}
			
			int items = concatenateUnsignedBytes(file[4], file[5], file[6], file[7]);
			int rows = concatenateUnsignedBytes(file[8], file[9], file[10], file[11]);
			int cols = concatenateUnsignedBytes(file[12], file[13], file[14], file[15]);
			
			int offset = 16;
			
			data = new Vector[items];
			
			for (int i = 0; i < items; i++) {
				double[] image = new double[rows * cols];
				
				for (int pixel = 0; pixel < (rows * cols); pixel++) {
					image[pixel] = unsigned(file[(pixel + offset + (i * rows * cols))]) > 127 ? 1.5 : -1.5;
				}
				
				data[i] = new Vector(image);
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return data;
	}
	
	/**
	 * Loads an MNIST image label file. The labels are individual digits corresponding to the correct handwritten digits in an image file.
	 * 
	 * @param labelPath path to MNIST label file
	 * @return MNIST labels
	 */
	public static final byte[] loadLabels(final String labelPath) {
		byte[] data = null;
		
		try {
			byte[] file = Files.readAllBytes(Paths.get(labelPath));
			int magicNum = concatenateUnsignedBytes(file[0], file[1], file[2], file[3]);
			
			if (magicNum != LABEL_MAGIC_NUMBER) {
				System.out.println("MNIST Label magic number is not equal to expected magic number!");
			}
			
			int items = concatenateUnsignedBytes(file[4], file[5], file[6], file[7]);
			
			int offset = 8;
			data = new byte[items];
			for (int i = 0; i < items; i++) {
				data[i] = file[i + offset];
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return data;
	}
	
	/**
	 * Concatenates unsigned bytes into an int.
	 * 
	 * @param b0 first byte
	 * @param b1 second byte
	 * @param b2 third byte
	 * @param b3 fourth byte
	 * @return (first + second + third + fourth), where + is a concatenation operator
	 */
	private static final int concatenateUnsignedBytes(final byte b0, final byte b1, final byte b2, final byte b3) {
		return (unsigned(b0) << 24) | (unsigned(b1) << 16) | (unsigned(b2) << 8) | unsigned(b3);
	}
	
	/**
	 * Converts a byte into an unsigned byte, stored in an int.
	 * 
	 * @param b byte
	 * @return int
	 */
	private static final int unsigned(final byte b) {
		return b & 0xFF;
	}
}
