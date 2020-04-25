package dezzy.neuronz2.arch.test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import dezzy.neuronz2.ann.error.VectorErrorFunctions;
import dezzy.neuronz2.ann.layers.DenseLayer;
import dezzy.neuronz2.arch.ForwardPassResult;
import dezzy.neuronz2.arch.LayeredNetwork;
import dezzy.neuronz2.arch.init.WeightInitFunc;
import dezzy.neuronz2.arch.layers.ElementActivationLayer;
import dezzy.neuronz2.arch.layers.Layer;
import dezzy.neuronz2.arch.layers.LayerSequence;
import dezzy.neuronz2.dataio.MnistLoader;
import dezzy.neuronz2.math.constructs.FuncDerivPair;
import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.constructs.Vector;
import dezzy.neuronz2.math.utility.IndexedGenerator;
import dezzy.neuronz2.math.utility.TensorApplier;

/**
 * Some functions to test the new architecture, and ensure that neural networks built with layers
 * from the new architecture can learn correctly.
 *
 * @author Joe Desmond
 */
@SuppressWarnings("unused")
public class NewNetworkTest {
	
	/**
	 * Runs one or more of the test functions defined in this class.
	 * 
	 * @param args unused
	 * @throws IOException if there is a problem saving/loading network files
	 * @throws ClassNotFoundException if there is a problem deserializing networks from files
	 */
	public static final void main(final String[] args) throws IOException, ClassNotFoundException {
		//andGateTest();
		//lrnLoadTest();
		//andGateTest2();
		mnistANNTest();
	}
	
	/**
	 * Tests the MNIST handwritten digit dataset on a fully connected neural network with 1 hidden layer
	 * consisting of 30 neurons. The neural network is constructed using the new layer architecture.
	 * 
	 * @throws IOException if there is a problem saving the network
	 */
	private static final void mnistANNTest() throws IOException {
		
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
		
		for (int i = 0; i < testDigits.length; i++) {
			final double[] values = new double[10];
			values[testDigits[i]] = 1;
			testOutputs[i] = new Vector(values);
		}
		
		final InputOutputPair[] trainingData = new InputOutputPair[trainingInputs.length];
		for (int i = 0; i < trainingInputs.length; i++) {
			trainingData[i] = new InputOutputPair(trainingInputs[i], trainingOutputs[i]);
		}
		
		final InputOutputPair[] testData = new InputOutputPair[testInputs.length];
		for (int i = 0; i < testInputs.length; i++) {
			testData[i] = new InputOutputPair(testInputs[i], testOutputs[i]);
		}
		
		final Random random = new Random();
		
		final DenseLayer inputLayer = DenseLayer.generate(random, WeightInitFunc.STANDARD_NORMAL_INIT, WeightInitFunc.STANDARD_NORMAL_INIT, 784, 30);
		final ElementActivationLayer<Vector> sigmoid0 = new ElementActivationLayer<>(FuncDerivPair.SIGMOID);
		final DenseLayer hiddenLayer = DenseLayer.generate(random, WeightInitFunc.XAVIER_INIT, WeightInitFunc.XAVIER_INIT, 30, 10);
		final ElementActivationLayer<Vector> sigmoid1 = new ElementActivationLayer<>(FuncDerivPair.SIGMOID);
		
		final List<Layer<Vector, Vector>> layers = List.of(inputLayer, sigmoid0, hiddenLayer, sigmoid1);
		
		final Layer<Vector, Vector> layerSequence = new LayerSequence<>(layers);
		final LayeredNetwork<Vector, Vector> network = new LayeredNetwork<>(layerSequence, VectorErrorFunctions.MEAN_SQUARE_ERROR);
		System.out.println("Network initialized with " + layerSequence.parameterCount() + " learnable parameters.");
		
		final int minibatchSize = 10;
		final double learningRate = 2.0;
		
		for (int epoch = 0; epoch < 30; epoch++) {
			Collections.shuffle(Arrays.asList(trainingData));
			
			//Training images			
			int index = 0;
			while (index < trainingData.length) {
				
				for (int i = 0; i < minibatchSize; i++) {
					final Vector input = trainingData[index].input;
					final Vector expectedOutput = trainingData[index].output;
					
					final ForwardPassResult<Vector> result = network.forwardPass(input, expectedOutput);
					network.backprop(expectedOutput, result.actualOutput, result.error);
					
					index++;
				}
				
				network.update(learningRate / minibatchSize);
			}
			
			//Test images
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
					
					//System.out.println(j + "\t" + expectedOutput.get(j) + "\t" + actual.get(j));
				
				if (expectedOutput.get(greatestIndex) == 1) {
					correct++;
				}
			}
			
			System.out.println("Epoch " + epoch + ": " + correct + "/" + testData.length);
			network.saveAs("networks/new-arch-test/mnist-ann.lrn");
		}
	}
	
	private static final class InputOutputPair {
		
		/**
		 * Input vector
		 */
		final Vector input;
		
		/**
		 * Expected output vector
		 */
		final Vector output;
		
		/**
		 * Creates an {@link InputOutputPair} with the given input and expected output vectors.
		 * 
		 * @param _input input vector
		 * @param _output expected output vector
		 */
		InputOutputPair(final Vector _input, final Vector _output) {
			input = _input;
			output = _output;
		}
	}
	
	/**
	 * Tests the and-gate network using better weight initialization strategies.
	 * @throws IOException if the network cannot be saved
	 */
	private static final void andGateTest2() throws IOException {
		final Random random = new Random();
		
		final DenseLayer layer0 = DenseLayer.generate(random, WeightInitFunc.XAVIER_INIT, WeightInitFunc.ZERO_INIT, 2, 4);
		final ElementActivationLayer<Vector> layer1 = new ElementActivationLayer<>(FuncDerivPair.SIGMOID);
		final DenseLayer layer2 = DenseLayer.generate(random, WeightInitFunc.XAVIER_INIT, WeightInitFunc.ZERO_INIT, 4, 1);
		final ElementActivationLayer<Vector> layer3 = new ElementActivationLayer<>(FuncDerivPair.SIGMOID);
		
		final List<Layer<Vector, Vector>> layers = List.of(layer0, layer1, layer2, layer3);
		final Layer<Vector, Vector> networkLayers = new LayerSequence<Vector>(layers);
		final LayeredNetwork<Vector, Vector> network = new LayeredNetwork<>(networkLayers, VectorErrorFunctions.MEAN_SQUARE_ERROR);
		
		final Vector input0 = new Vector(0, 0);
		final Vector input1 = new Vector(0, 1);
		final Vector input2 = new Vector(1, 0);
		final Vector input3 = new Vector(1, 1);
		
		final Vector exp0 = new Vector(0);
		final Vector exp1 = new Vector(1);
		
		final Vector[] inputs = {input0, input1, input2, input3};
		final Vector[] expecteds = {exp0, exp0, exp0, exp1};
		
		
		for (int i = 0; i < 10; i++) {
			int successes = 0;
			
			for (int j = 0; j < 100; j++) {
				final int index = (int)(Math.random() * 4);
				final Vector input = inputs[index];
				final Vector expected = expecteds[index];
				
				final ForwardPassResult<Vector> result = network.forwardPass(input, expected);
				
				final int expValue = (int)expected.get(0);
				final Vector actual = result.actualOutput;
				final double actualValue = actual.get(0);
				
				if ((actualValue < 0.5 && expValue == 0) || (actualValue >= 0.5 && expValue == 1)) {
					successes++;
				}
				
				network.backprop(expected, actual, result.error);
				network.update(1);
			}
			
			System.out.println(i + ":\t" + successes + "/100");
		}
		
		System.out.println("andgate2.lrn");
		System.out.println(network.network.parameterCount() + " learnable parameters");
		//LayeRed Network, or LeaRN
		network.saveAs("networks/new-arch-test/andgate2.lrn");
	}
	
	/**
	 * Tries to load the "and-gate" network saved by {@link #andGateTest()} and test it
	 * on some inputs.
	 * 
	 * @throws IOException if there is a problem reading the file
	 * @throws ClassNotFoundException if there is a problem deserializing the object stored in the file
	 */
	private static final void lrnLoadTest() throws ClassNotFoundException, IOException {
		final LayeredNetwork<Vector, Vector> andNetwork = LayeredNetwork.loadFrom("networks/new-arch-test/andgate.lrn");
		final TensorApplier<Vector> func = v -> new Vector(Math.round(v.get(0)));
		final Vector in0 = new Vector(0, 0);
		final Vector in1 = new Vector(0, 1);
		final Vector in2 = new Vector(1, 0);
		final Vector in3 = new Vector(1, 1);
		
		final Vector[] inputs = {in0, in1, in2, in3};
		
		for (Vector in : inputs) {
			final Vector out = andNetwork.network.forwardPass(in);
			System.out.println(in + " -> " + func.apply(out).get(0) + "  (" + out.get(0) + ")");
		}
	}
	
	/**
	 * A test network implementing a logical "and" gate.
	 * @throws IOException if the network cannot be saved
	 */
	private static final void andGateTest() throws IOException {
		final IndexedGenerator generator = (indices) -> {
			return Math.random() + 0.1;
		};
		
		final Matrix layer0w = Matrix.generate(generator, 4, 2);
		final Matrix layer1w = Matrix.generate(generator, 1, 4);
		
		final Vector layer0b = Vector.generate(generator, 4);
		final Vector layer1b = Vector.generate(generator, 1);
		
		final DenseLayer layer0 = new DenseLayer(layer0w, layer0b);
		final DenseLayer layer1 = new DenseLayer(layer1w, layer1b);
		final ElementActivationLayer<Vector> sigmoid0 = new ElementActivationLayer<>(FuncDerivPair.SIGMOID);
		final ElementActivationLayer<Vector> sigmoid1 = new ElementActivationLayer<>(FuncDerivPair.SIGMOID);
		
		List<Layer<Vector, Vector>> layers = new ArrayList<>();
		layers.add(layer0);
		layers.add(sigmoid0);
		layers.add(layer1);
		layers.add(sigmoid1);
		
		
		final LayerSequence<Vector> sequence = new LayerSequence<Vector>(layers);
		final LayeredNetwork<Vector, Vector> network = new LayeredNetwork<>(sequence, VectorErrorFunctions.MEAN_SQUARE_ERROR);
		
		final Vector input0 = new Vector(0, 0);
		final Vector input1 = new Vector(0, 1);
		final Vector input2 = new Vector(1, 0);
		final Vector input3 = new Vector(1, 1);
		
		final Vector exp0 = new Vector(0);
		final Vector exp1 = new Vector(1);
		
		final Vector[] inputs = {input0, input1, input2, input3};
		final Vector[] expecteds = {exp0, exp0, exp0, exp1};
		
		
		for (int i = 0; i < 10; i++) {
			int successes = 0;
			
			for (int j = 0; j < 100; j++) {
				final int index = (int)(Math.random() * 4);
				final Vector input = inputs[index];
				final Vector expected = expecteds[index];
				
				final ForwardPassResult<Vector> result = network.forwardPass(input, expected);
				
				final int expValue = (int)expected.get(0);
				final Vector actual = result.actualOutput;
				final double actualValue = actual.get(0);
				
				if ((actualValue < 0.5 && expValue == 0) || (actualValue >= 0.5 && expValue == 1)) {
					successes++;
				}
				
				network.backprop(expected, actual, result.error);
				network.update(1);
			}
			
			System.out.println(i + ":\t" + successes + "/100");
		}
		
		//LayeRed Network, or LeaRN
		network.saveAs("networks/new-arch-test/andgate.lrn");
	}
}
