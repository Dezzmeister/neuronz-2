package dezzy.neuronz2.arch.layers;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import dezzy.neuronz2.arch.ParallelBackwardPass;
import dezzy.neuronz2.arch.ParallelForwardPass;
import dezzy.neuronz2.arch.ParallelLayer;
import dezzy.neuronz2.math.constructs.ElementContainer;

/**
 * An implementation of {@link LayerSequence} for parallel layers.
 *
 * @author Joe Desmond
 * @param <T> tensor type of this layer sequence
 * @see LayerSequence
 */
public class ParallelLayerSequence<T extends ElementContainer<T>> implements ParallelLayer<T, T> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8378506635729469407L;
	
	/**
	 * The layers in this sequence
	 */
	private final List<ParallelLayer<T, T>> layers;
	
	/**
	 * Constructs a ParallelLayerSequence with the given layers. The layers are connected in the order
	 * that they are provided.
	 * 
	 * @param _layers sequence of layers
	 */
	public ParallelLayerSequence(final List<ParallelLayer<T, T>> _layers) {
		layers = _layers;
	}
	
	/**
	 * Propagates the input through every layer in this sequence.
	 * 
	 * @param prevActivations the input to this layer sequence, or the output of a previous layer
	 * @return activations for the next layer
	 */
	@Override
	public T forwardPass(final T prevActivations) {
		T activations = prevActivations;
		
		for (int i = 0; i < layers.size(); i++) {
			activations = layers.get(i).forwardPass(activations);
		}
		
		return activations;
	}

	/**
	 * Propagates the error gradient through every layer in this sequence. Every layer except the first
	 * this sequence is called with <code>isFirstLayer</code> set to false, and the value of 
	 * <code>isFirstLayer</code> given to this function is passed into the first layer of the sequence.
	 * 
	 * @param errorOutputDeriv (partial) derivative of the network error with respect to this layer's output
	 * @param isFirstLayer true if this is the first layer in the network
	 * @return (partial) derivative of the network error with respect to this layer's input
	 */
	@Override
	public T backprop(final T errorOutputDeriv, final boolean isFirstLayer) {
		T derivative = errorOutputDeriv;
		
		for (int i = layers.size() - 1; i >= 1; i--) {
			derivative = layers.get(i).backprop(derivative, false);
		}
		
		return layers.get(0).backprop(derivative, isFirstLayer);
	}

	/**
	 * Calls {@link #update(double) update(learningRate)} for every layer in this sequence, starting with the first.
	 * Some layers may not use this function.
	 * 
	 * @param learningRate the learning rate (for gradient descent)
	 */
	@Override
	public void update(final double learningRate) {
		for (int i = 0; i < layers.size(); i++) {
			layers.get(i).update(learningRate);
		}
	}

	/**
	 * Returns the number of learnable parameters in this layer sequence, which is the sum of all the learnable parameters
	 * in the {@linkplain #layers layer list}.
	 * 
	 * @return total number of learnable parameters in this layer sequence
	 */
	@Override
	public int parameterCount() {
		int sum = 0;
		
		for (int i = 0; i < layers.size(); i++) {
			sum += layers.get(i).parameterCount();
		}
		
		return sum;
	}
	
	/**
	 * Returns the total number of sub-layers in this layer sequence, which is the sum of all the sub-layers
	 * in the {@linkplain #layers layer list}.
	 * 
	 * @return total number of sub-layers in this layer sequence
	 */
	@Override
	public int sublayers() {
		int sublayerCount = 0;
		
		for (int i = 0; i < layers.size(); i++) {
			sublayerCount += layers.get(i).sublayers();
		}
		
		return sublayerCount;
	}

	@Override
	public ParallelForwardPass<T> parallelForwardPass(final T prevActivations) {
		final Map<Layer<?, ?>, ElementContainer<?>> latestInputs = new HashMap<>();
		final Map<Layer<?, ?>, ElementContainer<?>> latestOutputs = new HashMap<>();
		
		ParallelForwardPass<T> activations = new ParallelForwardPass<>(prevActivations, Map.of(), Map.of());
		
		for (int i = 0; i < layers.size(); i++) {
			activations = layers.get(i).parallelForwardPass(activations.output);
			latestInputs.putAll(activations.latestInputs);
			latestOutputs.putAll(activations.latestOutputs);
		}
		
		final T output = activations.output;
		
		return new ParallelForwardPass<>(output, latestInputs, latestOutputs);
	}

	@Override
	public ParallelBackwardPass<T> parallelBackprop(final ParallelForwardPass<T> prevForward, final T errorOutputDeriv, final boolean isFirstLayer) {
		final Map<Layer<?, ?>, List<ElementContainer<?>>> gradients = new HashMap<>();
		
		ParallelBackwardPass<T> derivative = new ParallelBackwardPass<>(errorOutputDeriv, Map.of());
		
		for (int i = layers.size() - 1; i >= 1; i--) {
			derivative = layers.get(i).parallelBackprop(prevForward, derivative.errorInputDeriv, false);
			gradients.putAll(derivative.gradients);
		}
		
		final ParallelBackwardPass<T> out = layers.get(0).parallelBackprop(prevForward, derivative.errorInputDeriv, isFirstLayer);
		gradients.putAll(out.gradients);
		
		return new ParallelBackwardPass<>(out.errorInputDeriv, gradients);
	}

	@Override
	public void parallelUpdate(final ParallelBackwardPass<?> gradients, final double learningRate) {
		for (int i = 0; i < layers.size(); i++) {
			layers.get(i).parallelUpdate(gradients, learningRate);
		}
	}	
}
