package dezzy.neuronz2.russianness;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import javax.imageio.ImageIO;

import dezzy.neuronz2.ann.error.VectorErrorFunctions;
import dezzy.neuronz2.ann.layers.DenseLayer;
import dezzy.neuronz2.ann.layers.SoftmaxLayer;
import dezzy.neuronz2.arch.ParallelBackwardPass;
import dezzy.neuronz2.arch.ParallelForwardPass;
import dezzy.neuronz2.arch.ParallelLayer;
import dezzy.neuronz2.arch.ParallelNetworkPass;
import dezzy.neuronz2.arch.init.WeightInitFunc;
import dezzy.neuronz2.arch.layers.ElementActivationLayer;
import dezzy.neuronz2.arch.layers.Layer;
import dezzy.neuronz2.arch.layers.ParallelLayerSequence;
import dezzy.neuronz2.cnn.ParallelConvNet;
import dezzy.neuronz2.cnn.layers.ConvFlattener;
import dezzy.neuronz2.cnn.layers.ConvolutionLayer2;
import dezzy.neuronz2.cnn.layers.PoolingLayer;
import dezzy.neuronz2.cnn.pooling.PoolingOperation;
import dezzy.neuronz2.math.constructs.ElementContainer;
import dezzy.neuronz2.math.constructs.FuncDerivPair;
import dezzy.neuronz2.math.constructs.Tensor3;
import dezzy.neuronz2.math.constructs.Vector;

/**
 * Tests related to the Russianness project.
 *
 * @author Joe Desmond
 */
@SuppressWarnings("unused")
public class RussiannessTest {
	
	private static final int IMG_DIMENSION = 256;
	
	public static final void main(final String[] args) throws IOException, ClassNotFoundException, InterruptedException, ExecutionException {
		russiannessTest();		
	}
	
	private static final void loadAndSavePaintings() throws IOException {
		final RussianDataset dataset = loadImages("data/russianness/labels.txt", "data/russianness/paintings/");
		dataset.saveAs("data/russianness/paintingdata.dat");
	}
	
	private static final RussianDataset loadImages(final String labelPath, final String baseImagePath) throws IOException {
		final List<String> labelFile = Files.readAllLines(Paths.get(labelPath));
		
		int index = 0;
		while (index < labelFile.size() && !labelFile.get(index).startsWith("Data[")) {
			index++;
		}
		
		final int numPaintings = Integer.parseInt(labelFile.get(index).substring(5, labelFile.get(index).indexOf("]")));
		index++;
		
		final Tensor3[] out = new Tensor3[numPaintings];
		final double[] russiannesses = new double[numPaintings];
		
		for (int i = 0; i < numPaintings; i++) {
			final String[] parts = labelFile.get(i + index).split("\t");
			
			final double russianness = Double.parseDouble(parts[1]);
			russiannesses[i] = russianness;
			
			final BufferedImage inputImage = ImageIO.read(new FileInputStream(new File(baseImagePath + parts[0])));
			System.out.println((inputImage == null));
			System.out.println(i + "\t" + inputImage.getType());
			final BufferedImage outputImage = new BufferedImage(IMG_DIMENSION, IMG_DIMENSION, inputImage.getType());
			final Graphics2D g2d = outputImage.createGraphics();			
			g2d.drawImage(inputImage, 0, 0, IMG_DIMENSION, IMG_DIMENSION, null);
			g2d.dispose();
			
			final double[][][] layers = new double[3][IMG_DIMENSION][IMG_DIMENSION];
			
			for (int y = 0; y < IMG_DIMENSION; y++) {
				for (int x = 0; x < IMG_DIMENSION; x++) {
					final int color = outputImage.getRGB(x, y);
					final double blue = ((color & 0xFF) / (double)255.0) - 0.5;
					final double green = (((color >> 8) & 0xFF) / (double)255.0) - 0.5;
					final double red = (((color >> 16) & 0xFF) / (double)255.0) - 0.5;
					
					layers[0][y][x] = red;
					layers[1][y][x] = green;
					layers[2][y][x] = blue;
				}
			}
			
			out[i] = new Tensor3(layers);
		}
		
		return new RussianDataset(out, russiannesses);
	}
	
	/**
	 * Encodes Russianness as a one-hot vector with length five. Component zero is
	 * "not Russian," and component 4 is "definitely Russian."
	 * 
	 * @param russianness russianness double value: one of 0.00, 0.25, 0.50, 0.75, or 1.00
	 * @return one-hot encoded Russianness
	 */
	private static final Vector oneHotRussianness(final double russianness) {		
		if (russianness == 0.00) {
			return new Vector(1, 0, 0, 0, 0);
		} else if (russianness == 0.25) {
			return new Vector(0, 1, 0, 0, 0);
		} else if (russianness == 0.50) {
			return new Vector(0, 0, 1, 0, 0);
		} else if (russianness == 0.75) {
			return new Vector(0, 0, 0, 1, 0);
		} else {
			return new Vector(0, 0, 0, 0, 1);
		}
	}
	
	@SuppressWarnings("unchecked")
	private static final void russiannessTest() throws ClassNotFoundException, IOException, InterruptedException, ExecutionException {
		final RussianDataset dataset = RussianDataset.loadFrom("data/russianness/paintingdata.dat");
		System.out.println("Loaded the Russianness dataset, containing " + dataset.images.length + " paintings");
		
		//Create the Russianness network
		
		/**
		 * Network structure (16 layers):
		 * 
		 * conv0 -> relu0 -> maxpooling0
		 * conv1 -> relu1 -> maxpooling1
		 * conv2 -> relu2 -> maxpooling2
		 * flatten
		 * fc0 -> sigmoid0
		 * fc1 -> sigmoid1
		 * fc2 -> softmax
		 */
		final Random random = new Random();
		
		final ConvolutionLayer2 conv0 = ConvolutionLayer2.generate(random, WeightInitFunc.KAIMING_INIT, WeightInitFunc.STANDARD_NORMAL_INIT, 5, 3, 5, 5);
		final ElementActivationLayer<Tensor3> relu0 = new ElementActivationLayer<>(FuncDerivPair.LEAKY_RELU);
		final PoolingLayer maxpooling0 = new PoolingLayer(PoolingOperation.MAX_POOLING, 2, 2, 2, 2);
		final ConvolutionLayer2 conv1 = ConvolutionLayer2.generate(random, WeightInitFunc.KAIMING_INIT, WeightInitFunc.STANDARD_NORMAL_INIT, 10, 5, 5, 5);
		final ElementActivationLayer<Tensor3> relu1 = new ElementActivationLayer<>(FuncDerivPair.LEAKY_RELU);
		final PoolingLayer maxpooling1 = new PoolingLayer(PoolingOperation.MAX_POOLING, 2, 2, 2, 2);
		final ConvolutionLayer2 conv2 = ConvolutionLayer2.generate(random, WeightInitFunc.KAIMING_INIT, WeightInitFunc.STANDARD_NORMAL_INIT, 10, 10, 5, 5);
		final ElementActivationLayer<Tensor3> relu2 = new ElementActivationLayer<>(FuncDerivPair.LEAKY_RELU);
		final PoolingLayer maxpooling2 = new PoolingLayer(PoolingOperation.MAX_POOLING, 5, 5, 5, 5);
		
		final List<ParallelLayer<Tensor3, Tensor3>> featureExtractorLayers = List.of(conv0, relu0, maxpooling0, conv1, relu1, maxpooling1, conv2, relu2, maxpooling2);
		final ParallelLayer<Tensor3, Tensor3> featureExtractor = new ParallelLayerSequence<>(featureExtractorLayers);
		final ParallelLayer<Tensor3, Vector> flattener = new ConvFlattener(10, 11, 11);
		
		final DenseLayer fc0 = DenseLayer.generate(random, WeightInitFunc.STANDARD_NORMAL_INIT, WeightInitFunc.STANDARD_NORMAL_INIT, 1210, 500);
		final ElementActivationLayer<Vector> sigmoid0 = new ElementActivationLayer<>(FuncDerivPair.SIGMOID);
		final DenseLayer fc1 = DenseLayer.generate(random, WeightInitFunc.STANDARD_NORMAL_INIT, WeightInitFunc.STANDARD_NORMAL_INIT, 500, 100);
		final ElementActivationLayer<Vector> sigmoid1 = new ElementActivationLayer<>(FuncDerivPair.SIGMOID);
		final DenseLayer fc2 = DenseLayer.generate(random, WeightInitFunc.STANDARD_NORMAL_INIT, WeightInitFunc.STANDARD_NORMAL_INIT, 100, 5);
		final SoftmaxLayer softmax = new SoftmaxLayer();
		
		final List<ParallelLayer<Vector, Vector>> classifierLayers = List.of(fc0, sigmoid0, fc1, sigmoid1, fc2, softmax);
		final ParallelLayer<Vector, Vector> classifier = new ParallelLayerSequence<>(classifierLayers);
		
		final ParallelConvNet<Tensor3, Vector> convNetwork = new ParallelConvNet<>(featureExtractor, flattener, classifier);
		
		System.out.println("Initialized neural network with " + convNetwork.parameterCount() + " learnable parameters and " + convNetwork.sublayers() + " layers");
		
		
		//Create one-hot expected data vectors
		final Vector[] expectedOutputs = new Vector[dataset.images.length];
		
		for (int i = 0; i < expectedOutputs.length; i++) {
			final double russianness = dataset.russiannesses[i];
			expectedOutputs[i] = oneHotRussianness(russianness);
		}
		
		final double learningRate = 2.0;
		final int minibatchSize = 10;		
		final ExecutorService threadPool = Executors.newFixedThreadPool(minibatchSize);
		
		for (int epoch = 0; epoch < dataset.images.length; epoch++) {
			
			int successes = 0;
			
			for (int i = 0; i < dataset.images.length; i += minibatchSize) {
				final ParallelNetworkPass<Tensor3, Vector>[] passes = (ParallelNetworkPass<Tensor3, Vector>[]) new ParallelNetworkPass<?, ?>[minibatchSize];
				final Future<ParallelBackwardPass<Tensor3>>[] minibatchResults = (Future<ParallelBackwardPass<Tensor3>>[]) new Future<?>[minibatchSize];
				
				//Submit training jobs
				for (int j = 0; j < minibatchSize; j++) {
					final ParallelNetworkPass<Tensor3, Vector> networkPass = new ParallelNetworkPass<>(convNetwork, dataset.images[i + j], expectedOutputs[i + j], VectorErrorFunctions.CROSS_ENTROPY);
					final Future<ParallelBackwardPass<Tensor3>> future = threadPool.submit(networkPass);
					passes[j] = networkPass;
					minibatchResults[j] = future;
				}
				
				// Sum all the gradients here
				final Map<Layer<?,?>, List<ElementContainer<?>>> gradients = new HashMap<>();
				
				//Evaluate network and propagate gradients
				for (int j = 0; j < minibatchSize; j++) {
					final ParallelNetworkPass<Tensor3, Vector> networkPass = passes[j];
					final ParallelBackwardPass<Tensor3> backpass = minibatchResults[j].get();
					
					final Vector actual = networkPass.actualOutput;
					final Vector expected = networkPass.expectedOutput;
					
					int greatestIndex = 0;
					double greatestValue = Double.NEGATIVE_INFINITY;
					
					for (int k = 0; k < actual.dimension; k++) {
						final double currentValue = actual.get(k);
						
						if (currentValue > greatestValue) {
							greatestIndex = k;
							greatestValue = currentValue;
						}
					}
					
					if (expected.get(greatestIndex) == 1) {
						successes++;
					}
					
					final Map<Layer<?,?>, List<ElementContainer<?>>> currentGradients = backpass.gradients;
					final Set<Layer<?, ?>> layerSet = currentGradients.keySet();
					
					for (final Layer<?, ?> layer : layerSet) {
						if (gradients.containsKey(layer)) {
							final List<ElementContainer<?>> currentValue = currentGradients.get(layer);
							
							final List<ElementContainer<?>> mapValue = gradients.get(layer);
							
							gradients.put(layer, sumGradients(mapValue, currentValue));
						} else {
							final List<ElementContainer<?>> currentValue = currentGradients.get(layer);
							
							if (currentValue != null) {								
								final List<ElementContainer<?>> mapValue = gradients.get(layer);
								
								if (mapValue != null) {
									gradients.put(layer, sumGradients(mapValue, currentValue));
								} else {
									gradients.put(layer, currentValue);
								}
							}
						}
					}
				}
				
				convNetwork.parallelUpdate(new ParallelBackwardPass<Tensor3>(null, gradients), learningRate);
			}
			
			System.out.println("Epoch " + epoch + ": " + successes + "/" + dataset.images.length);
		}
	}
	
	private static final List<ElementContainer<?>> sumGradients(final List<ElementContainer<?>> g0, final List<ElementContainer<?>> g1) {
		final List<ElementContainer<?>> out = new ArrayList<>();
		
		for (int i = 0; i < g0.size(); i++) {
			final ElementContainer<?> e0 = g0.get(i);
			final ElementContainer<?> e1 = g1.get(i);
			final ElementContainer<?> e2 = (ElementContainer<?>) e0.unsafePlus(e1);
			out.add(e2);
		}
		
		return out;
	}
	
	/**
	 * This tests the new parallel layer architecture, which will be used for the Russianness project
	 * to accelerate learning.
	 */
	private static final void parallelArchitectureTest() {
		final Random random = new Random();
		
		final ConvolutionLayer2 conv0 = ConvolutionLayer2.generate(random, WeightInitFunc.KAIMING_INIT, WeightInitFunc.STANDARD_NORMAL_INIT, 5, 3, 5, 5);
		final ElementActivationLayer<Tensor3> relu0 = new ElementActivationLayer<>(FuncDerivPair.LEAKY_RELU);
		final PoolingLayer maxpooling0 = new PoolingLayer(PoolingOperation.MAX_POOLING, 2, 2, 2, 2);
		final ConvolutionLayer2 conv1 = ConvolutionLayer2.generate(random, WeightInitFunc.KAIMING_INIT, WeightInitFunc.STANDARD_NORMAL_INIT, 10, 5, 5, 5);
		final ElementActivationLayer<Tensor3> relu1 = new ElementActivationLayer<>(FuncDerivPair.LEAKY_RELU);
		final PoolingLayer maxpooling1 = new PoolingLayer(PoolingOperation.MAX_POOLING, 2, 2, 2, 2);
		final ConvolutionLayer2 conv2 = ConvolutionLayer2.generate(random, WeightInitFunc.KAIMING_INIT, WeightInitFunc.STANDARD_NORMAL_INIT, 10, 10, 5, 5);
		final ElementActivationLayer<Tensor3> relu2 = new ElementActivationLayer<>(FuncDerivPair.LEAKY_RELU);
		final PoolingLayer maxpooling2 = new PoolingLayer(PoolingOperation.MAX_POOLING, 5, 5, 5, 5);
		
		final List<ParallelLayer<Tensor3, Tensor3>> featureExtractorLayers = List.of(conv0, relu0, maxpooling0, conv1, relu1, maxpooling1, conv2, relu2, maxpooling2);
		final ParallelLayer<Tensor3, Tensor3> featureExtractor = new ParallelLayerSequence<>(featureExtractorLayers);
		final ParallelLayer<Tensor3, Vector> flattener = new ConvFlattener(10, 11, 11);
		
		final DenseLayer fc0 = DenseLayer.generate(random, WeightInitFunc.STANDARD_NORMAL_INIT, WeightInitFunc.STANDARD_NORMAL_INIT, 1210, 500);
		final ElementActivationLayer<Vector> sigmoid0 = new ElementActivationLayer<>(FuncDerivPair.SIGMOID);
		final DenseLayer fc1 = DenseLayer.generate(random, WeightInitFunc.STANDARD_NORMAL_INIT, WeightInitFunc.STANDARD_NORMAL_INIT, 500, 100);
		final ElementActivationLayer<Vector> sigmoid1 = new ElementActivationLayer<>(FuncDerivPair.SIGMOID);
		final DenseLayer fc2 = DenseLayer.generate(random, WeightInitFunc.STANDARD_NORMAL_INIT, WeightInitFunc.STANDARD_NORMAL_INIT, 100, 10);
		final SoftmaxLayer softmax = new SoftmaxLayer();
		
		final List<ParallelLayer<Vector, Vector>> classifierLayers = List.of(fc0, sigmoid0, fc1, sigmoid1, fc2, softmax);
		final ParallelLayer<Vector, Vector> classifier = new ParallelLayerSequence<>(classifierLayers);
		
		final ParallelConvNet<Tensor3, Vector> convNetwork = new ParallelConvNet<>(featureExtractor, flattener, classifier);
		//final LayeredNetwork<Tensor3, Vector> network = new LayeredNetwork<>(convNetwork, VectorErrorFunctions.CROSS_ENTROPY);
		
		System.out.println("Initialized neural network with " + convNetwork.parameterCount() + " learnable parameters.");
		
		final Tensor3 input = Tensor3.generate(i -> 0, 3, 256, 256);
		
		final ParallelForwardPass<Vector> forwardResult = convNetwork.parallelForwardPass(input);
		
		//final Vector output = convNetwork.forwardPass(input);
		System.out.println(forwardResult.output.shape());
		
		final ParallelBackwardPass<Tensor3> backwardResult = convNetwork.parallelBackprop(forwardResult, forwardResult.output, true);
		
		//Expecting null
		System.out.println(backwardResult.errorInputDeriv);
	}
}
