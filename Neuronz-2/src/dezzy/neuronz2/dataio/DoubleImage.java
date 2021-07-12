package dezzy.neuronz2.dataio;

import java.io.IOException;

import dezzy.neuronz2.dataio.image.IntImage;
import dezzy.neuronz2.math.constructs.Vector;

/**
 * A grayscale image represented by normalized doubles. Can be converted to a vector and
 * used in a neural network.
 * 
 * @author Joe Desmond
 */
public class DoubleImage extends IntImage {
	
	/**
	 * Normalized grayscale pixels - 0.0 maps to 0 and 1.0 maps to 255.
	 * {@link IntImage#pixels} contains full ARGB pixels.
	 */
	public final double[] normPixels;
	
	/**
	 * Loads an image from a file, converts it to grayscale, and stores the grayscale pixels in {@link #normPixels}.
	 * 
	 * @param filename path to image
	 * @throws IOException if there is a problem loading the image
	 */
	public DoubleImage(final String filename) throws IOException {
		super(filename);
		normPixels = normalize(pixels);
	}
	
	/**
	 * Constructs an image from existing normalized pixels and image dimensions.
	 * 
	 * @param _normPixels normalized pixels, 0.0 is 0 and 1.0 is 255
	 * @param _width width of the image in pixels
	 * @param _height height of the image in pixels
	 */
	public DoubleImage(final double[] _normPixels, final int _width, final int _height) {
		super(getARGBPixels(_normPixels), _width, _height);
		normPixels = _normPixels;
	}
	
	/**
	 * Convert normalized pixels to grayscale ARGB pixels.
	 * 
	 * @param normPixels normalized pixels
	 * @return ARGB pixels
	 */
	private static int[] getARGBPixels(final double[] normPixels) {
		final int[] out = new int[normPixels.length];
		
		for (int i = 0; i < normPixels.length; i++) {
			final int pix = (int)(normPixels[i] * 255);
			final int pixel = (0xFF << 24) | (pix << 16) | (pix << 8) | pix;
			
			out[i] = pixel;
		}
		
		return out;
	}
	
	/**
	 * Convert RGB (or ARGB) pixels to normalized grayscale pixels.
	 * 
	 * @param rgb RGB or ARGB pixels
	 * @return normalized grayscale pixels
	 */
	private double[] normalize(final int[] rgb) {
		final double[] out = new double[rgb.length];
		
		for (int i = 0; i < rgb.length; i++) {
			final int pixel = rgb[i];
			final int blue  = (pixel & 0xFF);
			final int green = (pixel >>> 8) & 0xFF;
			final int red = (pixel >>> 16) & 0xFF;
			
			final int avg = (red + green + blue) / 3;
			out[i] = avg / 255.0;
		}
		
		return out;
	}
	
	/**
	 * Convert the normalized pixels to a Vector.
	 * 
	 * @return a Vector with the normalized pixels as components
	 */
	public Vector toVector() {
		return new Vector(normPixels);
	}
}
