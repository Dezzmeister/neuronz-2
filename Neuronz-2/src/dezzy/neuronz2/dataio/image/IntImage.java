package dezzy.neuronz2.dataio.image;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import dezzy.neuronz2.math.utility.DimensionMismatchException;

/**
 * An image with a width and a height. Pixels are laid out row-wise.
 * 
 * @author Joe Desmond
 */
public class IntImage {
	
	/**
	 * Row-wise ARGB pixels
	 */
	public final int[] pixels;
	
	/**
	 * Width of the image in pixels
	 */
	public final int width;
	
	/**
	 * Height of the image in pixels
	 */
	public final int height;
	
	/**
	 * Loads an image from a file.
	 * 
	 * @param filename image file
	 * @throws IOException if there is a problem loading the file
	 */
	public IntImage(final String filename) throws IOException {
		final BufferedImage img = ImageIO.read(new File(filename));
		width = img.getWidth();
		height = img.getHeight();
		pixels = new int[width * height];
		
		img.getRGB(0, 0, width, height, pixels, 0, width);
	}
	
	/**
	 * Constructs an image from an existing pixel array.
	 * 
	 * @param _pixels pixel array
	 * @param _width width of the image in pixels
	 * @param _height height of the image in pixels
	 */
	public IntImage(final int[] _pixels, final int _width, final int _height) {
		if (_pixels.length != _width * _height) {
			throw new DimensionMismatchException("Pixel array length needs to match product of width and height");
		}
		
		pixels = _pixels;
		width = _width;
		height = _height;
	}
	
	/**
	 * Saves the image to a file in a given format. If this method is called by a subclass, the image
	 * may be saved in grayscale.
	 * 
	 * @param filename path to output file
	 * @param formatName file format string: "PNG", "JPG", etc.
	 * @throws IOException if there is a problem saving the file
	 */
	public void save(final String filename, final String formatName) throws IOException {
		final BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
		img.setRGB(0, 0, width, height, pixels, 0, width);
		
		ImageIO.write(img, formatName, new File(filename));
	}
}
