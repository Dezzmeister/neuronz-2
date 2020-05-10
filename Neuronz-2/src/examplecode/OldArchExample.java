package examplecode;

import java.io.IOException;
import java.util.concurrent.ExecutionException;

import dezzy.neuronz2.dataio.MnistLoader;
import dezzy.neuronz2.math.constructs.FuncDerivPair;
import dezzy.neuronz2.math.constructs.Vector;
import dezzy.neuronz2.math.utility.OutputVerificationScheme;
import dezzy.neuronz2.network.LearningRateAdjuster;
import dezzy.neuronz2.network.Network;
import dezzy.neuronz2.network.NetworkRunner;
import dezzy.neuronz2.network.ProcessingScheme;

/**
 * Some example code demonstrating the use of the old architecture on the MNIST handwritten digit data.
 *
 * @author Joe Desmond
 */
public class OldArchExample {
	
	/**
	 * A test demonstrating a neural network classifying handwritten digits. The network is constructed using the old
	 * Neuronz-2 architecture.
	 * 
	 * @param args unused
	 * @throws ClassNotFoundException if there is a problem loading a network from a file
	 * @throws IOException if there is a problem reading from or writing to a file
	 */
	public static final void main(final String[] args) throws ClassNotFoundException, IOException {
		
		// Load the input images as flattened 784-component vectors. There are 60000 training images
		final Vector[] inputImages = MnistLoader.loadImages("data/mnist/train-images.idx3-ubyte");
		
		// Load the expected digits (60000 expected digits)
		final byte[] expectedDigits = MnistLoader.loadLabels("data/mnist/train-labels.idx1-ubyte");
		final Vector[] idealOutputs = new Vector[inputImages.length];
		
		// Construct one-hot expected output vectors
		for (int i = 0; i < expectedDigits.length; i++) {
			final double[] values = new double[10];
			values[expectedDigits[i]] = 1;
			idealOutputs[i] = new Vector(values);
		}
		
		// Load 10000 test images
		final Vector[] testImages = MnistLoader.loadImages("data/mnist/test-images.idx3-ubyte");
		
		// Load 10000 expected digits
		final byte[] testDigits = MnistLoader.loadLabels("data/mnist/test-labels.idx1-ubyte");
		final Vector[] idealTestOutputs = new Vector[testImages.length];
		
		// Construct one-hot expected output vectors
		for (int i = 0; i < testDigits.length; i++) {
			final double[] values = new double[10];
			values[testDigits[i]] = 1;
			idealTestOutputs[i] = new Vector(values);
		}
		
		/**
		 * Create a network with two hidden layers using the sigmoid function between each layer using the MSE cost function.
		 * One of the flaws of the old architecture is that it couples the network with the cost function used to train it.
		 */
		final Network network = new Network(new int[] {784, 100, 50, 10}, new FuncDerivPair[] {FuncDerivPair.SIGMOID, FuncDerivPair.SIGMOID, FuncDerivPair.SIGMOID}, Network.MSE_DERIV);
		System.out.println("Creating a deep neural network with 784 input neurons, 100 hidden neurons in the first hidden layer, 50 hidden neurons in the second hidden layer, and 10 output neurons; using sigmoid for all layers and MSE as the cost function.");
		
		// Networks can be loaded from files using the static method Network.loadFrom()
		//final Network network = Network.loadFrom("networks/mnist/networks/mnist/example/old-arch-example.ntwk2");
		
		
		/**
		 * A learning rate adjuster function to decrease the learning rate as the success rate increases.
		 */
		final LearningRateAdjuster learningRateSchedule = (lr, epoch, success) -> {
			if (success <= 0.85) {
				return 1.9;
			} else if (success <= 0.9) {
				return 1.0;
			} else if (success <= 0.93) {
				return 0.4;
			} else {
				return 0.3;
			}
		};
		
		/**
		 * Returns true if the actual output matches the expected (ideal) output. Evaluates the success of one pass
		 * through the network.
		 */
		final OutputVerificationScheme evaluator = (actual, ideal) -> {
			int greatestIndex = 0;
			double greatestValue = 0;
			for (int i = 0; i < 10; i++) {
				double currentValue = actual.get(i);
				
				if (currentValue > greatestValue) {
					greatestValue = currentValue;
					greatestIndex = i;
				}
			}
			
			return ideal.get(greatestIndex) == 1;
		};
		
		/**
		 * Create a NetworkRunner to train the network on the given training and test data.
		 */
		final NetworkRunner networkRunner = new NetworkRunner(network, inputImages, idealOutputs, testImages, idealTestOutputs, false);
		
		try {
			
			/**
			 * Train the network over 30 epochs using minibatch SGD with a batch size of 10. After each epoch, the accuracy of the network
			 * is evaluated on the test data and the most accurate network is saved to the given file. The network will be trained using a 
			 * multithreaded CPU scheme employing a threadpool to split each batch.
			 */
			networkRunner.run(30, 10, learningRateSchedule, evaluator, "networks/mnist/example/old-arch-example.ntwk2", ProcessingScheme.CPU_MULTITHREADED);
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (ExecutionException e) {
			e.printStackTrace();
		}
	}
}
