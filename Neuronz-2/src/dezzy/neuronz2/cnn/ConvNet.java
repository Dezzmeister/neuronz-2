package dezzy.neuronz2.cnn;

import dezzy.neuronz2.arch.layers.Layer;
import dezzy.neuronz2.math.constructs.ElementContainer;

/**
 * A convolutional neural network with a feature extractor, flattening layer, and classifier.
 *
 * @author Joe Desmond
 * @param <I> input tensor type (usually {@link dezzy.neuronz2.math.constructs.Tensor3 Tensor3})
 * @param <O> output tensor type (usually {@link dezzy.neuronz2.math.constructs.Vector Vector})
 */
public class ConvNet<I extends ElementContainer<I>, O extends ElementContainer<O>> implements Layer<I, O> {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -5208939238932619194L;
	
	/**
	 * Feature extractor (usually convolutional layers, pooling, etc.)
	 */
	private final Layer<I, I> featureExtractor;
	
	/**
	 * Flattening layer, converts output from feature extractor into suitable input for classifier
	 */
	private final Layer<I, O> flattener;
	
	/**
	 * Classifier (usually a fully connected network, implemented with
	 * {@linkplain dezzy.neuronz2.ann.layers.DenseLayer dense layers}
	 */
	private final Layer<O, O> classifier;
	
	/**
	 * Creates a convolutional neural network from the given components, which may also be smaller networks in themselves.
	 * 
	 * @param _featureExtractor feature extractor
	 * @param _flattener flattening layer
	 * @param _classifier classifier
	 */
	public ConvNet(final Layer<I, I> _featureExtractor, final Layer<I, O> _flattener, final Layer<O, O> _classifier) {
		featureExtractor = _featureExtractor;
		flattener = _flattener;
		classifier = _classifier;
	}

	@Override
	public O forwardPass(final I prevActivations) {
		final I featureMaps = featureExtractor.forwardPass(prevActivations);
		final O flattenedFeatures = flattener.forwardPass(featureMaps);
		final O finalOutput = classifier.forwardPass(flattenedFeatures);
		
		return finalOutput;
	}

	@Override
	public I backprop(final O errorOutputDeriv, final boolean isFirstLayer) {
		final O classifierDeriv = classifier.backprop(errorOutputDeriv, false);
		final I unflattenedDeriv = flattener.backprop(classifierDeriv, false);
		final I errorInputDeriv = featureExtractor.backprop(unflattenedDeriv, isFirstLayer);
		
		return errorInputDeriv;
	}

	@Override
	public void update(final double learningRate) {
		classifier.update(learningRate);
		flattener.update(learningRate);
		featureExtractor.update(learningRate);
	}
	
	/**
	 * Adds the total number of learnable parameters in the three component layers of this
	 * convolutional neural network.
	 * 
	 * @return number of learnable parameters in this network
	 */
	@Override
	public int parameterCount() {
		return featureExtractor.parameterCount() + flattener.parameterCount() + classifier.parameterCount();
	}
	
	/**
	 * Returns the sum of the sublayers in the {@linkplain #featureExtractor feature extractor},
	 * {@linkplain #flattener flattener}, and {@linkplain #classifier classifier}.
	 * 
	 * @return total number of sublayers in this layer
	 */
	@Override
	public int sublayers() {
		return featureExtractor.sublayers() + flattener.sublayers() + classifier.sublayers();
	}
}
