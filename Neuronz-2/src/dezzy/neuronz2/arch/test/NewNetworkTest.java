package dezzy.neuronz2.arch.test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.zip.DataFormatException;

import dezzy.neuronz2.ann.error.VectorErrorFunctions;
import dezzy.neuronz2.ann.layers.DenseLayer;
import dezzy.neuronz2.ann.layers.SoftmaxLayer;
import dezzy.neuronz2.arch.ForwardPassResult;
import dezzy.neuronz2.arch.LayeredNetwork;
import dezzy.neuronz2.arch.init.WeightInitFunc;
import dezzy.neuronz2.arch.layers.ElementActivationLayer;
import dezzy.neuronz2.arch.layers.Layer;
import dezzy.neuronz2.arch.layers.LayerSequence;
import dezzy.neuronz2.cnn.layers.ConvolutionLayer2;
import dezzy.neuronz2.cnn.layers.PoolingLayer;
import dezzy.neuronz2.cnn.pooling.PoolingOperation;
import dezzy.neuronz2.dataio.MnistConvLoader;
import dezzy.neuronz2.dataio.MnistLoader;
import dezzy.neuronz2.math.constructs.ElementContainer;
import dezzy.neuronz2.math.constructs.FuncDerivPair;
import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.constructs.Tensor3;
import dezzy.neuronz2.math.constructs.Vector;
import dezzy.neuronz2.math.utility.IndexedGenerator;
import dezzy.neuronz2.math.utility.TensorApplier;

/**
 * Some functions to test the new architecture, and ensure that neural networks built with layers
 * from the new architecture can learn correctly.
 *
 * @author Joe Desmond
 */
@SuppressWarnings({"unused", "unchecked"})
public class NewNetworkTest {
	
	/**
	 * Runs one or more of the test functions defined in this class.
	 * 
	 * @param args unused
	 * @throws IOException if there is a problem saving/loading files
	 * @throws ClassNotFoundException if there is a problem deserializing networks from files
	 */
	public static final void main(final String[] args) throws IOException, ClassNotFoundException {
		//andGateTest();
		//lrnLoadTest();
		//andGateTest2();
		//mnistANNTest();
		//mnistANNSoftmaxTest();
		leNetSizeTest();
	}
	
	private static final void leNetSizeTest() {
		final Random random = new Random();
		
		final ConvolutionLayer2 conv0 = ConvolutionLayer2.generate(random, WeightInitFunc.STANDARD_NORMAL_INIT, WeightInitFunc.STANDARD_NORMAL_INIT, 20, 1, 5, 5);
		final ElementActivationLayer<Tensor3> relu0 = new ElementActivationLayer<>(FuncDerivPair.LEAKY_RELU);
		final PoolingLayer maxpooling0 = new PoolingLayer(PoolingOperation.MAX_POOLING, 2, 2, 2, 2);
		final ConvolutionLayer2 conv1 = ConvolutionLayer2.generate(random, WeightInitFunc.STANDARD_NORMAL_INIT, WeightInitFunc.STANDARD_NORMAL_INIT, 30, 20, 5, 5);
		final ElementActivationLayer<Tensor3> relu1 = new ElementActivationLayer<>(FuncDerivPair.LEAKY_RELU);
		final PoolingLayer maxpooling1 = new PoolingLayer(PoolingOperation.MAX_POOLING, 2, 2, 2, 2);
		
		final List<Layer<Tensor3, Tensor3>> layers = List.of(conv0, relu0, maxpooling0, conv1, relu1, maxpooling1);
		
		final Layer<Tensor3, Tensor3> testLayers = new LayerSequence<>(layers);
		
		final Tensor3 input = Tensor3.generate(i -> 0, 1, 28, 28);
		
		final Tensor3 output = testLayers.forwardPass(input);
		
		System.out.println(output.dimension);
	}
	
	private static final void mnistCNNTest() throws IOException, DataFormatException {
		
		/**
		 * 28 x 28 x 1 tensors, 28 pixels wide by 28 pixels tall, 1 layer deep for grayscale image (training images)
		 */
		final Tensor3[] trainingInputs = MnistConvLoader.loadImages("data/mnist/train-images.idx3-ubyte");
		
		/**
		 * Expected digits for each MNIST training image
		 */
		final byte[] trainingDigits = MnistLoader.loadLabels("data/mnist/train-labels.idx1-ubyte");
		
		/**
		 * One-hot expected training digit vectors
		 */
		final Vector[] trainingOutputs = new Vector[trainingInputs.length];
		
		for (int i = 0; i < trainingDigits.length; i++) {
			final double[] values = new double[10];
			values[trainingDigits[i]] = 1;
			trainingOutputs[i] = new Vector(values);
		}
		
		/**
		 * 28 x 28 x 1 tensors, 28 pixels wide by 28 pixels tall, 1 layer deep for grayscale image (test images)
		 */
		final Tensor3[] testInputs = MnistConvLoader.loadImages("data/mnist/test-images.idx3-ubyte");
		
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
		
		final InputOutputPair<Tensor3, Vector>[] trainingData = (InputOutputPair<Tensor3, Vector>[]) new InputOutputPair<?, ?>[trainingInputs.length];
		for (int i = 0; i < trainingInputs.length; i++) {
			trainingData[i] = new InputOutputPair<>(trainingInputs[i], trainingOutputs[i]);
		}
		
		final InputOutputPair<Tensor3, Vector>[] testData = (InputOutputPair<Tensor3, Vector>[]) new InputOutputPair<?, ?>[testInputs.length];
		for (int i = 0; i < testInputs.length; i++) {
			testData[i] = new InputOutputPair<>(testInputs[i], testOutputs[i]);
		}
		
		final Random random = new Random();
		
		// TODO: finish this
	}
	
	/**
	 * Same as {@link #mnistANNTest()}, except the softmax activation function is used as the last layer (instead of sigmoid)
	 * and the network is trained with cross-entropy instead of mean-square-error.
	 * 
	 * @throws IOException if there is a problem saving the network
	 */
	private static final void mnistANNSoftmaxTest() throws IOException {
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
		
		final InputOutputPair<Vector, Vector>[] trainingData = (InputOutputPair<Vector, Vector>[]) new InputOutputPair<?, ?>[trainingInputs.length];
		for (int i = 0; i < trainingInputs.length; i++) {
			trainingData[i] = new InputOutputPair<>(trainingInputs[i], trainingOutputs[i]);
		}
		
		final InputOutputPair<Vector, Vector>[] testData = (InputOutputPair<Vector, Vector>[]) new InputOutputPair<?, ?>[testInputs.length];
		for (int i = 0; i < testInputs.length; i++) {
			testData[i] = new InputOutputPair<>(testInputs[i], testOutputs[i]);
		}
		
		final Random random = new Random();
		
		final DenseLayer inputLayer = DenseLayer.generate(random, WeightInitFunc.STANDARD_NORMAL_INIT, WeightInitFunc.STANDARD_NORMAL_INIT, 784, 30);
		final ElementActivationLayer<Vector> sigmoid0 = new ElementActivationLayer<>(FuncDerivPair.SIGMOID);
		final DenseLayer hiddenLayer = DenseLayer.generate(random, WeightInitFunc.XAVIER_INIT, WeightInitFunc.XAVIER_INIT, 30, 10);
		final SoftmaxLayer softmax = new SoftmaxLayer();
		
		final List<Layer<Vector, Vector>> layers = List.of(inputLayer, sigmoid0, hiddenLayer, softmax);
		
		final Layer<Vector, Vector> layerSequence = new LayerSequence<>(layers);
		final LayeredNetwork<Vector, Vector> network = new LayeredNetwork<>(layerSequence, VectorErrorFunctions.CROSS_ENTROPY);
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
				
				if (expectedOutput.get(greatestIndex) == 1) {
					correct++;
				}
			}
			
			System.out.println("Epoch " + epoch + ": " + correct + "/" + testData.length);
			network.saveAs("networks/new-arch-test/mnist-ann-softmax.lrn");
		}
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
		
		final InputOutputPair<Vector, Vector>[] trainingData = (InputOutputPair<Vector, Vector>[]) new InputOutputPair<?, ?>[trainingInputs.length];
		for (int i = 0; i < trainingInputs.length; i++) {
			trainingData[i] = new InputOutputPair<>(trainingInputs[i], trainingOutputs[i]);
		}
		
		final InputOutputPair<Vector, Vector>[] testData = (InputOutputPair<Vector, Vector>[]) new InputOutputPair<?, ?>[testInputs.length];
		for (int i = 0; i < testInputs.length; i++) {
			testData[i] = new InputOutputPair<>(testInputs[i], testOutputs[i]);
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
				
				if (expectedOutput.get(greatestIndex) == 1) {
					correct++;
				}
			}
			
			System.out.println("Epoch " + epoch + ": " + correct + "/" + testData.length);
			network.saveAs("networks/new-arch-test/mnist-ann.lrn");
		}
	}
	
	/**
	 * An input to a neural network and the expected output, for that input.
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
