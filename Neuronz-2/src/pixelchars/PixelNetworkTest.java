package pixelchars;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;

import javax.imageio.ImageIO;

import dezzy.neuronz2.math.constructs.Vector;
import dezzy.neuronz2.math.utility.OutputVerificationScheme;
import dezzy.neuronz2.network.LearningRateAdjuster;
import dezzy.neuronz2.network.Network;
import dezzy.neuronz2.network.NetworkRunner;
import dezzy.neuronz2.network.ProcessingScheme;



/**
 * Tests a network that classifies a 5x5 image containing a capital letter of the alphabet.
 *
 * @author Joe Desmond
 */
@SuppressWarnings("unused")
public final class PixelNetworkTest {
	private static final Map<Character, Vector> inputMap = new HashMap<Character, Vector>();
	
	public static void main(final String[] args) throws IOException, InterruptedException, ExecutionException, ClassNotFoundException {
		populateMap();
		testExistingNetwork();		
	}
	
	private static final void testExistingNetwork() throws ClassNotFoundException, IOException {
		final Network network = Network.loadFrom("networks/pixelchars/deep-network-100percent.ntwk");
		final int tests = 100;
		
		for (int i = 0; i < tests; i++) {
			final char letter = (char)((int)(Math.random() * 26) + 'A');
			final Vector input = inputMap.get(letter);
			final Vector[] activations = network.run(input);
			final Vector output = network.getLatestOutput(activations);
			
			int greatestIndex = 0;
			double greatestValue = 0;
			for (int j = 0; j < output.dimension; j++) {
				if (output.get(j) > greatestValue) {
					greatestValue = output.get(j);
					greatestIndex = j;
				}
			}
			
			final char predicted = (char)(greatestIndex + 'A');
			System.out.println(letter + "\t" + predicted);
		}
	}
	
	private static final void trainNetwork() throws InterruptedException, ExecutionException {
		final Network network = new Network(25, 125, 50, 25, 12, 26);
		
		final int dataSize = 50000;
		final Vector[] inputs = new Vector[dataSize];
		final Vector[] idealOutputs = new Vector[dataSize];
		
		for (int i = 0; i < dataSize; i++) {
			final char letter = (char)((int)(Math.random() * 26) + 'A');
			final Vector input = inputMap.get(letter);
			final Vector idealOutput = buildIdealOutput(letter);
			inputs[i] = input;
			idealOutputs[i] = idealOutput;
		}
		
		final NetworkRunner runner = new NetworkRunner(network, inputs, idealOutputs, inputs, idealOutputs, false);
		
		final LearningRateAdjuster adjuster = (current, epoch, prevSuccess) -> {
			return 3.0;
		};
		
		final OutputVerificationScheme successMetric = OutputVerificationScheme.greatestOutputMetric;
		
		runner.run(30, 10, adjuster, successMetric, "networks/pixelchars/network-100h.ntwk", ProcessingScheme.CPU_MULTITHREADED);
	}
	
	private static final void populateMap() throws IOException {
		for (int i = 0; i < 26; i++) {
			final char letter = (char)(i + 'A');
			
			final int[] pixels = loadPixels("data/pixelchars/" + letter + ".png");
			final double[] inputs = new double[pixels.length];
			
			for (int j = 0; j < pixels.length; j++) {
				if (pixels[j] == 0) {
					inputs[j] = 0;
				} else {
					inputs[j] = 1.0;
				}
			}
			
			inputMap.put(letter, new Vector(inputs));
		}
	}	
	
	private static final Vector buildIdealOutput(final char c) {
		final int index = c - 'A';
		final double[] elements = new double[26];
		elements[index] = 1;
		
		return new Vector(elements);
	}
	
	
	
	/**
	 * Returns a list of ints corresponding to pixels in the image. A bitmask is applied to the raw
	 * rgb values, so that white pixels will be a 0 and black pixels will be a 1.
	 * 
	 * @param fileName	image file
	 * @return	array of pixels (0 if white, 1 if black)
	 * @throws IOException if there is a problem loading the image file
	 */
	private static final int[] loadPixels(final String fileName) throws IOException {
		final File imageFile = new File(fileName);
		final BufferedImage image = ImageIO.read(imageFile);
		final int[] rgbArray = new int[image.getWidth() * image.getHeight()];
		
		image.getRGB(0, 0, image.getWidth(), image.getHeight(), rgbArray, 0, image.getWidth());
		
		for (int i = 0; i < rgbArray.length; i++) {
			rgbArray[i] = ~rgbArray[i] & 0x1;		//0 if white, 1 if black
		}
		
		return rgbArray;
	}
}
