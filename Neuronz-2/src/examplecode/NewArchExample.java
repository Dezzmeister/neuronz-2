package examplecode;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import dezzy.neuronz2.ann.error.VectorErrorFunctions;
import dezzy.neuronz2.ann.layers.DenseLayer;
import dezzy.neuronz2.ann.layers.SoftmaxLayer;
import dezzy.neuronz2.arch.ForwardPassResult;
import dezzy.neuronz2.arch.LayeredNetwork;
import dezzy.neuronz2.arch.init.WeightInitFunc;
import dezzy.neuronz2.arch.layers.ElementActivationLayer;
import dezzy.neuronz2.arch.layers.Layer;
import dezzy.neuronz2.arch.layers.LayerSequence;
import dezzy.neuronz2.dataio.MnistLoader;
import dezzy.neuronz2.math.constructs.ElementContainer;
import dezzy.neuronz2.math.constructs.FuncDerivPair;
import dezzy.neuronz2.math.constructs.Vector;

/**
 * Example code demonstrating the use of the new architecture on the MNIST handwritten digit data.
 * This code does not demonstrate the parallel architecture. An example of the parallel architecture does
 * not exist yet, but code making use of it can be found {@linkplain dezzy.neuronz2.russianness.RussiannessTest here}.
 *
 * @author Joe Desmond
 */
public class NewArchExample {
	
	/**
	 * An input to a neural network and the expected output, for that input.
	 * This class is used to couple input/output pairs so that data can be shuffled
	 * between epochs.
	 *
	 * @author Joe Desmond
	 * @param <I> input tensor type
	 * @param <O> output tensor type
	 */
	private static final class InputOutputPair<I extends ElementContainer<I>, O extends ElementContainer<O>> {
		
		/**
		 * Input tensor
		 */
		final I input;
		
		/**
		 * Expected output tensor
		 */
		final O output;
		
		/**
		 * Creates an {@link InputOutputPair} with the given input and expected output tensors.
		 * 
		 * @param _input input tensor
		 * @param _output expected output tensor
		 */
		InputOutputPair(final I _input, final O _output) {
			input = _input;
			output = _output;
		}
	}
	
	/**
	 * Trains a fully connected network using the softmax activation function.
	 * 
	 * @param args unused
	 * @throws IOException if there is a problem saving the network
	 * @throws ClassNotFoundException  if there is a problem loading an existing network
	 */
	@SuppressWarnings("unchecked")
	public static final void main(final String[] args) throws IOException, ClassNotFoundException {
		/**
		 * Vectors of length 784, MNIST training image data
		 */
		final Vector[] trainingInputs = MnistLoader.loadImages("data/mnist/train-images.idx3-ubyte");
		
		/**
		 * Expected digits for each MNIST training image
		 */
		final byte[] expectedDigits = MnistLoader.loadLabels("data/mnist/train-labels.idx1-ubyte");
		
		/**
		 * One-hot expected training digit vectors
		 */
		final Vector[] trainingOutputs = new Vector[trainingInputs.length];
		
		// Construct the one-hot training output vectors
		for (int i = 0; i < expectedDigits.length; i++) {
			final double[] values = new double[10];
			values[expectedDigits[i]] = 1;
			trainingOutputs[i] = new Vector(values);
		}
		
		/**
		 * Vectors of length 784, MNIST test image data
		 */
		final Vector[] testInputs = MnistLoader.loadImages("data/mnist/test-images.idx3-ubyte");
		
		/**
		 * Expected digits for each MNIST test image
		 */
		final byte[] testDigits = MnistLoader.loadLabels("data/mnist/test-labels.idx1-ubyte");
		
		/**
		 * One-hot expected test digit vectors
		 */
		final Vector[] testOutputs = new Vector[testInputs.length];
		
		// Construct the one-hot test output vectors
		for (int i = 0; i < testDigits.length; i++) {
			final double[] values = new double[10];
			values[testDigits[i]] = 1;
			testOutputs[i] = new Vector(values);
		}
		
		// Group training inputs with their expected outputs
		final InputOutputPair<Vector, Vector>[] trainingData = (InputOutputPair<Vector, Vector>[]) new InputOutputPair<?, ?>[trainingInputs.length];
		for (int i = 0; i < trainingInputs.length; i++) {
			trainingData[i] = new InputOutputPair<>(trainingInputs[i], trainingOutputs[i]);
		}
		
		// Group test inputs with their expected outputs
		final InputOutputPair<Vector, Vector>[] testData = (InputOutputPair<Vector, Vector>[]) new InputOutputPair<?, ?>[testInputs.length];
		for (int i = 0; i < testInputs.length; i++) {
			testData[i] = new InputOutputPair<>(testInputs[i], testOutputs[i]);
		}
		
		// Random number generator
		final Random random = new Random();
		
		/**
		 * These are the layers of the network. The network is structured like this:
		 * 
		 * 		dense -> sigmoid -> dense -> softmax
		 * 
		 * The first dense layer accepts an input vector with 784 components and outputs
		 * a vector with 30 components. The hidden layer outputs a vector with 10 components,
		 * which is fed into the final softmax layer.
		 */
		final DenseLayer inputLayer = DenseLayer.generate(random, WeightInitFunc.STANDARD_NORMAL_INIT, WeightInitFunc.STANDARD_NORMAL_INIT, 784, 30);
		final ElementActivationLayer<Vector> sigmoid0 = new ElementActivationLayer<>(FuncDerivPair.SIGMOID);
		final DenseLayer hiddenLayer = DenseLayer.generate(random, WeightInitFunc.XAVIER_INIT, WeightInitFunc.XAVIER_INIT, 30, 10);
		final SoftmaxLayer softmax = new SoftmaxLayer();
		
		// Put each layer into a list to create a LayerSequence
		final List<Layer<Vector, Vector>> layers = List.of(inputLayer, sigmoid0, hiddenLayer, softmax);
		
		// Create a LayerSequence containing the entire network. This LayerSequence is also a single Layer
		final Layer<Vector, Vector> layerSequence = new LayerSequence<>(layers);
		
		// Create a network with the given LayerSequence. This constructor accepts a single Layer, because a Layer can have many sub-layers
		final LayeredNetwork<Vector, Vector> network = new LayeredNetwork<>(layerSequence, VectorErrorFunctions.CROSS_ENTROPY);
		System.out.println("Network initialized with " + layerSequence.parameterCount() + " learnable parameters.");
		
		// New-architecture networks can be loaded from a file using Layer.loadFrom()
		//final Layer<Vector, Vector> network = Layer.loadFrom("networks/mnist/example/new-arch-example.lrn");
		
		// Mini-batch size; run this many inputs before updating the network parameters
		final int minibatchSize = 10;
		
		// Static learning rate. A training framework for new-architecture networks does not exist yet, so training must be done manually
		final double learningRate = 2.0;
		
		for (int epoch = 0; epoch < 30; epoch++) {
			Collections.shuffle(Arrays.asList(trainingData));
			
			//Training images			
			int index = 0;
			while (index < trainingData.length) {
				
				// Run one mini-batch
				for (int i = 0; i < minibatchSize; i++) {
					final Vector input = trainingData[index].input;
					final Vector expectedOutput = trainingData[index].output;
					
					final ForwardPassResult<Vector> result = network.forwardPass(input, expectedOutput);
					network.backprop(expectedOutput, result.actualOutput, result.error);
					
					index++;
				}
				
				network.update(learningRate / minibatchSize);
			}
			
			//Test images. Gradients are not calculated here
			int correct = 0;
			for (int i = 0; i < testData.length; i++) {
				final Vector input = testData[i].input;
				final Vector expectedOutput = testData[i].output;
				
				final ForwardPassResult<Vector> result = network.forwardPass(input, expectedOutput);
				final Vector actual = result.actualOutput;
								
				int greatestIndex = 0;
				double greatestValue = 0;
				
				for (int j = 0; j < 10; j++) {
					double currentValue = actual.get(j);
					
					if (currentValue > greatestValue) {
						greatestValue = currentValue;
						greatestIndex = j;
					}
				}
				
				if (expectedOutput.get(greatestIndex) == 1) {
					correct++;
				}
			}
			
			System.out.println("Epoch " + epoch + ": " + correct + "/" + testData.length);
			network.saveAs("networks/mnist/example/new-arch-example.lrn");
		}
	}
}
