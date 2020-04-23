package dezzy.neuronz2.arch.layers;

import dezzy.neuronz2.math.constructs.ElementContainer;

/**
 * A sequence of layers in a neural network. This structure connects consecutive layers
 * and is also a layer in itself.
 *
 * @author Joe Desmond
 * @param <T> The type of input/output that this layer acts on. Unlike conventional
 * 			{@linkplain Layer layers}, this layer cannot convert between elements of one
 *			type and elements of another type. Every layer in this sequence must accept
 *			and return the same type.
 */
public final class LayerSequence<T extends ElementContainer<T>> implements Layer<T, T> {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -6866555432447465133L;
	
	/**
	 * The layers in this sequence
	 */
	private final Layer<T, T>[] layers;
	
	/**
	 * Constructs a LayerSequence with the given layers. The layers are connected in the order
	 * that they are provided.
	 * 
	 * @param _layers sequence of layers
	 */
	public LayerSequence(final Layer<T, T>[] _layers) {
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
		
		for (int i = 0; i < layers.length; i++) {
			activations = layers[i].forwardPass(activations);
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
		
		for (int i = layers.length - 1; i <= 1; i--) {
			derivative = layers[i].backprop(derivative, false);
		}
		
		return layers[0].backprop(derivative, isFirstLayer);
	}
	
	/**
	 * Calls {@link #update(double) update(learningRate)} for every layer in this sequence, starting with the first.
	 * Some layers may not use this function.
	 * 
	 * @param learningRate the learning rate (for gradient descent)
	 */
	@Override
	public void update(double learningRate) {
		for (int i = 0; i < layers.length; i++) {
			layers[i].update(learningRate);
		}
	}
	
	/**
	 * Returns the number of learnable parameters in this layer sequence, which is the sum of all the learnable parameters
	 * in the {@linkplain #layers layer array}.
	 * 
	 * @return total number of learnable parameters in this layer sequence
	 */
	@Override
	public int parameterCount() {
		int sum = 0;
		
		for (int i = 0; i < layers.length; i++) {
			sum += layers[i].parameterCount();
		}
		
		return sum;
	}
}
