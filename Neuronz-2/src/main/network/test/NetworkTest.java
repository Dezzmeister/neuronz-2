package main.network.test;

import java.awt.Transparency;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.ComponentColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.concurrent.ExecutionException;

import javax.imageio.ImageIO;

import dezzy.neuronz2.dataio.MnistLoader;
import dezzy.neuronz2.math.constructs.FuncDerivPair;
import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.constructs.Tensor3;
import dezzy.neuronz2.math.constructs.Vector;
import dezzy.neuronz2.math.utility.Functions;
import dezzy.neuronz2.math.utility.OutputVerificationScheme;
import dezzy.neuronz2.network.LearningRateAdjuster;
import dezzy.neuronz2.network.Network;
import dezzy.neuronz2.network.Network.BackpropPair;
import dezzy.neuronz2.network.NetworkFunctions;
import dezzy.neuronz2.network.NetworkRunner;
import dezzy.neuronz2.network.ProcessingScheme;


/**
 * Contains functions for testing neural networks.
 *
 * @author Joe Desmond
 */
@SuppressWarnings("unused")
public final class NetworkTest {
	
	public static final void main(final String[] args) throws ClassNotFoundException, IOException {
		//manualTest();
		//networkClassTest();
		//networkClassTest2();
		//mnistTest();
		mnistTest3();
		//digitTest();
	}
	
	private static final void digitTest() throws ClassNotFoundException, IOException {
		final Network network = Network.loadFrom("networks/mnist/network-100h-50h-2-3.ntwk2");
		final BufferedImage img = ImageIO.read(new File("digit.png"));
		final double[] pixels = new double[784];
		for (int y = 0; y < 28; y++) {
			for (int x = 0; x < 28; x++) {
				final int color = img.getRGB(x, y) & 0xFF;
				pixels[x + 28 * y] = (color < 127) ? 1.5 : -1.5;
			}
		}
		
		final Vector input = new Vector(pixels);
		final Vector[] outputs = network.run(input);
		final Vector output = network.getLatestOutput(outputs);
		
		int highestIndex = 0;
		for (int i = 0; i < 10; i++) {
			if (output.get(i) > output.get(highestIndex)) {
				highestIndex = i;
			}
		}
		System.out.println(highestIndex);
		System.out.println(output);
	}
	
	private static final void saveImage(final Vector pixels, final String name) {
		final byte[] imageData = new byte[pixels.dimension * 3];
		
		for (int i = 0; i < pixels.dimension; i++) {
			int pixel = (int)(pixels.get(i) * 255);
			
			imageData[i * 3] = (byte)pixel;
			imageData[i * 3 + 1] = (byte)pixel;
			imageData[i * 3 + 2] = (byte)pixel;
		}
		
		final DataBuffer buffer = new DataBufferByte(imageData, imageData.length);
		final WritableRaster raster = Raster.createInterleavedRaster(buffer, 28, 28, 3 * 28, 3, new int[] {0, 1, 2}, null);
		final ColorModel cm = new ComponentColorModel(ColorModel.getRGBdefault().getColorSpace(), false, true, Transparency.OPAQUE, DataBuffer.TYPE_BYTE);
		final BufferedImage image = new BufferedImage(cm, raster, true, null);
		
		try {
			ImageIO.write(image, "png", new File(name));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private static final void mnistTest3() throws ClassNotFoundException, IOException {
		final Vector[] inputImages = MnistLoader.loadImages("data/mnist/train-images.idx3-ubyte");
		
		for (int i = 0; i < inputImages.length; i++) {
			inputImages[i] = inputImages[i];
		}
		
		final byte[] expectedDigits = MnistLoader.loadLabels("data/mnist/train-labels.idx1-ubyte");
		final Vector[] idealOutputs = new Vector[inputImages.length];
		
		for (int i = 0; i < expectedDigits.length; i++) {
			final double[] values = new double[10];
			values[expectedDigits[i]] = 1;
			idealOutputs[i] = new Vector(values);
		}
		
		final ImageLabelPair[] data = new ImageLabelPair[50000];
		
		for (int i = 0; i < data.length; i++) {
			data[i] = new ImageLabelPair(inputImages[i], idealOutputs[i]);
		}
		
		final Vector[] testImages = MnistLoader.loadImages("data/mnist/test-images.idx3-ubyte");
		
		for (int i = 0; i < testImages.length; i++) {
			testImages[i] = testImages[i];
		}
		
		final byte[] testDigits = MnistLoader.loadLabels("data/mnist/test-labels.idx1-ubyte");
		final Vector[] idealTestOutputs = new Vector[testImages.length];
		
		for (int i = 0; i < testDigits.length; i++) {
			final double[] values = new double[10];
			values[testDigits[i]] = 1;
			idealTestOutputs[i] = new Vector(values);
		}
		
		final Network network = new Network(new int[] {784, 100, 50, 10}, new FuncDerivPair[] {FuncDerivPair.SIGMOID, FuncDerivPair.SIGMOID, FuncDerivPair.SIGMOID}, Network.MSE_DERIV);
		System.out.println("Creating a deep neural network with 784 input neurons, 100 hidden neurons in the first hidden layer, 50 hidden neurons in the second hidden layer, and 10 output neurons; using sigmoid for all layers and MSE as the cost function.");
		//final Network network = Network.loadFrom("networks/mnist/network-100h-50h-93percent.ntwk2");
		//final Network network = Network.loadFrom("networks/mnist/network-100h-50h-2-3.ntwk2");
		
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
		
		final NetworkRunner networkRunner = new NetworkRunner(network, inputImages, idealOutputs, testImages, idealTestOutputs, false);
		try {
			networkRunner.run(30, 10, learningRateSchedule, evaluator, "networks/mnist/network-100h-50h-alt-2.ntwk2", ProcessingScheme.CPU_MULTITHREADED);
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (ExecutionException e) {
			e.printStackTrace();
		}
	}
	
	private static final void mnistTest2() {
		final Vector[] inputImages = MnistLoader.loadImages("data/mnist/train-images.idx3-ubyte");
		
		for (int i = 0; i < inputImages.length; i++) {
			inputImages[i] = inputImages[i];
		}
		
		final byte[] expectedDigits = MnistLoader.loadLabels("data/mnist/train-labels.idx1-ubyte");
		final Vector[] idealOutputs = new Vector[inputImages.length];
		
		for (int i = 0; i < expectedDigits.length; i++) {
			final double[] values = new double[10];
			values[expectedDigits[i]] = 1;
			idealOutputs[i] = new Vector(values);
		}
		
		final ImageLabelPair[] data = new ImageLabelPair[50000];
		
		for (int i = 0; i < data.length; i++) {
			data[i] = new ImageLabelPair(inputImages[i], idealOutputs[i]);
		}
		
		final Vector[] testImages = MnistLoader.loadImages("data/mnist/test-images.idx3-ubyte");
		
		for (int i = 0; i < testImages.length; i++) {
			testImages[i] = testImages[i];
		}
		
		final byte[] testDigits = MnistLoader.loadLabels("data/mnist/test-labels.idx1-ubyte");
		
		final Network network = new Network(784, 100, 40, 10);
		
		double learningRate = 1.9f;
		final int miniBatchSize = 10;
		final int epochs = 30;
		
		for (int i = 0; i < epochs; i++) {
			Collections.shuffle(Arrays.asList(data));
			
			Tensor3 weightDeltas = network.weightTensor.minus(network.weightTensor);
			for (int j = 0; j < 50000 - miniBatchSize; j++) {
				final BackpropPair result = network.backprop(data[j].image, data[j].digit);
				Tensor3 deltas = result.weightDeltas;
				weightDeltas = weightDeltas.plus(deltas);
				
				if (j != 0 && j % miniBatchSize == 0) {
					weightDeltas = weightDeltas.transform(w -> w / (float)miniBatchSize);
					network.applyWeightDeltas(weightDeltas, learningRate);
					weightDeltas = network.weightTensor.minus(network.weightTensor);
				}
			}
			
			int successes = 0;
			for (int j = 0; j < 10000; j++) {
				final Vector output = network.getLatestOutput(network.run(testImages[j]));
				
				int highestDigit = 0;
				for (int k = 0; k < 10; k++) {
					if (output.get(k) > output.get(highestDigit)) {
						highestDigit = k;
					}
				}
				
				if (highestDigit == testDigits[j]) {
					successes++;
				}
			}
			
			System.out.println("Epoch " + i + ": " + successes + "/10000   " + ((100.0f * successes)/10000.0f) + "%");
			if (successes >= 8600) {
				learningRate = 1.1f;
			} 
		}
	}
	
	private static final class ImageLabelPair {
		final Vector image;
		final Vector digit;
		
		ImageLabelPair(final Vector _image, final Vector _digit) {
			image = _image;
			digit = _digit;
		}
	}
	
	private static final void networkClassTest() {
		final Matrix layer0 = new Matrix(new double[][] {
			{0.15f, 0.2f, 0.35f},
			{0.25f, 0.3f, 0.35f}
		});
		
		final Matrix layer1 = new Matrix(new double[][] {
			{0.4f, 0.45f, 0.6f},
			{0.50f, 0.55f, 0.6f}
		});
		
		final Tensor3 networkTensor = new Tensor3(layer0, layer1);
		final Network network = new Network(networkTensor);
		
		final Vector input = new Vector(0.05f, 0.1f, 1);
		final Vector ideal = new Vector(0.01f, 0.99f);
		final double eta = 0.5f;
		
		for (int i = 0; i < 1000000; i++) {
			final BackpropPair result = network.backprop(input, ideal);
			final Tensor3 deltas = result.weightDeltas;
			final Vector output = network.getLatestOutput(result.activations);
			network.applyWeightDeltas(deltas, eta);
			
			if (i % 100000 == 0) System.out.println(output);
		}
	}
	
	private static final void manualTest() {
		
		
		final Network network = new Network(2, 4, 3);
		
		final Matrix layer0 = network.weightTensor.getLayer(0);
		final Matrix layer1 = network.weightTensor.getLayer(1);
		
		/*
		final Matrix layer0 = new Matrix(new double[][] {
			{0.15f, 0.2f, 0.35f},
			{0.25f, 0.3f, 0.35f}
		});
		
		final Matrix layer1 = new Matrix(new double[][] {
			{0.4f, 0.45f, 0.6f},
			{0.50f, 0.55f, 0.6f}
		});
		*/
		
		final Tensor3 tensor0 = new Tensor3(layer0, layer1);
		System.out.println("Network tensor:\n" + tensor0 + "\n");
		
		final Vector activation0 = new Vector(0.05f, 0.1f, 1);
		System.out.println("First activation vector (layer 2): " + activation0);
		
		final Vector activation1 = NetworkFunctions.computeOutputVector(layer0, activation0, Functions::sigmoid).append(1); //Append 1 (for biases)
		System.out.println("Second activation vector (layer 1): " + activation1);
		
		final Vector activation2 = NetworkFunctions.computeOutputVector(layer1, activation1, Functions::sigmoid);
		System.out.println("Third activation vector (layer 0): " + activation2);
		
		final Vector ideal0 = new Vector(0.01f, 0.99f, 0.04f);
		System.out.println("Ideal output: " + ideal0);
		
		final double totalError0 = NetworkFunctions.computeTotalMSE(ideal0, activation2);
		System.out.println("Total error: " + totalError0);
		
		final Vector errorOutputDeriv = activation2.elementOperation(ideal0, (actual, ideal) -> - (ideal - actual));
		System.out.println("Partial derivative of error wrt output (layer 0): " + errorOutputDeriv);
		
		final Vector sigmoidDeriv0 = activation2.transform(out -> out * (1 - out)); //out/net
		System.out.println("Partial derivative of output wrt input (layer 0): " + sigmoidDeriv0);
		
		final Vector errorInputDeriv0 = errorOutputDeriv.hadamard(sigmoidDeriv0);
		
		final Matrix weightDeltas0 = errorInputDeriv0.outerProduct(activation1);
		System.out.println("Weight gradients (layer 0): \n" + weightDeltas0);
		
		final double eta0 = 0.5f;
		System.out.println("Learning rate: " + eta0);
		
		final Matrix newWeights0 = layer1.minus(weightDeltas0.transform(w -> w * eta0));
		System.out.println("New weights (layer 0): \n" + newWeights0);
		System.out.println();
		
		
		
		final Vector activationDeriv1 = activation1.transform(out -> out * (1 - out));
		System.out.println("Partial derivative of layer 1 outputs wrt layer 1 inputs: " + activationDeriv1);
		
		final Vector nextError0 = layer1.transpose().multiply(errorInputDeriv0);
		System.out.println("Partial derivative of error wrt output (layer 1): " + nextError0);
		
		final Vector sigmoidDeriv1 = activation1.transform(out -> out * (1 - out));
		System.out.println("Partial derivative of output wrt input (layer 1): " + sigmoidDeriv1);
		
		final Vector errorInputDeriv1 = nextError0.hadamard(sigmoidDeriv1);
		System.out.println("Partial derivative of error wrt input (layer 1): " + errorInputDeriv1);
		
		final Matrix weightDeltas1 = errorInputDeriv1.removeLastElement().outerProduct(activation0);
		System.out.println("Weight gradients (layer 1): \n" + weightDeltas1);
		
		final Matrix newWeights1 = layer0.minus(weightDeltas1.transform(w -> w * eta0));
		System.out.println("New weights (layer 1): \n" + newWeights1);
	}
}
