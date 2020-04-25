package dezzy.neuronz2.cl;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Utility functions for GPU optimizations with OpenCL.
 *
 * @author Joe Desmond
 */
public class CLUtilities {
	
	/**
	 * Loads the contents of a file into a single US-ASCII String.
	 * 
	 * @param path path to the file
	 * @return a string containing the contents of the file
	 * @throws IOException if there is a problem reading the file
	 */
	public static final String readString(final String path) throws IOException {
		final byte[] encoded = Files.readAllBytes(Paths.get(path));
		return new String(encoded, StandardCharsets.US_ASCII);
	}
}
