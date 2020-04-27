package dezzy.neuronz2.cnn;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import dezzy.neuronz2.arch.ParallelBackwardPass;
import dezzy.neuronz2.arch.ParallelForwardPass;
import dezzy.neuronz2.arch.ParallelLayer;
import dezzy.neuronz2.arch.layers.Layer;
import dezzy.neuronz2.math.constructs.ElementContainer;

public class ParallelConvNet<I extends ElementContainer<I>, O extends ElementContainer<O>> implements ParallelLayer<I, O> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4671589715078208307L;
	
	/**
	 * Feature extractor (usually convolutional layers, pooling, etc.)
	 */
	private final ParallelLayer<I, I> featureExtractor;
	
	/**
	 * Flattening layer, converts output from feature extractor into suitable input for classifier
	 */
	private final ParallelLayer<I, O> flattener;
	
	/**
	 * Classifier (usually a fully connected network, implemented with
	 * {@linkplain dezzy.neuronz2.ann.layers.DenseLayer dense layers}
	 */
	private final ParallelLayer<O, O> classifier;
	
	/**
	 * Creates a convolutional neural network from the given components, which may also be smaller networks in themselves.
	 * 
	 * @param _featureExtractor feature extractor
	 * @param _flattener flattening layer
	 * @param _classifier classifier
	 */
	public ParallelConvNet(final ParallelLayer<I, I> _featureExtractor, final ParallelLayer<I, O> _flattener, final ParallelLayer<O, O> _classifier) {
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

	@Override
	public ParallelForwardPass<O> parallelForwardPass(final I prevActivations) {
		final ParallelForwardPass<I> featureMaps = featureExtractor.parallelForwardPass(prevActivations);
		final ParallelForwardPass<O> flattenedFeatures = flattener.parallelForwardPass(featureMaps.output);
		final ParallelForwardPass<O> finalOutput = classifier.parallelForwardPass(flattenedFeatures.output);
		
		final Map<Layer<?, ?>, ElementContainer<?>> latestInputs = new HashMap<>();
		final Map<Layer<?, ?>, ElementContainer<?>> latestOutputs = new HashMap<>();
		
		latestInputs.putAll(featureMaps.latestInputs);
		latestInputs.putAll(flattenedFeatures.latestInputs);
		latestInputs.putAll(finalOutput.latestInputs);
		
		latestOutputs.putAll(featureMaps.latestOutputs);
		latestOutputs.putAll(flattenedFeatures.latestOutputs);
		latestOutputs.putAll(finalOutput.latestOutputs);
		
		return new ParallelForwardPass<>(finalOutput.output, latestInputs, latestOutputs);
	}

	@Override
	public ParallelBackwardPass<I> parallelBackprop(final ParallelForwardPass<O> prevForward, final O errorOutputDeriv, final boolean isFirstLayer) {
		final ParallelForwardPass<I> forwardI = new ParallelForwardPass<>(null, prevForward.latestInputs, prevForward.latestOutputs);
		
		final ParallelBackwardPass<O> classifierDeriv = classifier.parallelBackprop(prevForward, errorOutputDeriv, false);
		final ParallelBackwardPass<I> unflattenedDeriv = flattener.parallelBackprop(prevForward, classifierDeriv.errorInputDeriv, false);
		final ParallelBackwardPass<I> errorInputDeriv = featureExtractor.parallelBackprop(forwardI, unflattenedDeriv.errorInputDeriv, isFirstLayer);
		
		final Map<Layer<?, ?>, List<ElementContainer<?>>> gradients = new HashMap<>();
		
		gradients.putAll(classifierDeriv.gradients);
		gradients.putAll(unflattenedDeriv.gradients);
		gradients.putAll(errorInputDeriv.gradients);
		
		return new ParallelBackwardPass<>(errorInputDeriv.errorInputDeriv, gradients);
	}

	@Override
	public void parallelUpdate(final ParallelBackwardPass<?> gradients, final double learningRate) {
		classifier.parallelUpdate(gradients, learningRate);
		flattener.parallelUpdate(gradients, learningRate);
		featureExtractor.parallelUpdate(gradients, learningRate);
	}
	
}
