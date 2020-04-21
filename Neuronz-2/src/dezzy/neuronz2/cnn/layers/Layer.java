package dezzy.neuronz2.cnn.layers;

import dezzy.neuronz2.math.constructs.Tensor3;

/**
 * A single layer in a convolutional neural network.
 *
 * @author Joe Desmond
 */
public interface Layer {
	
	/**
	 * Get the activations for this layer (the result of forward propagation of the previous activations).
	 * This function may return a tensor with a different shape than the input tensor.
	 * 
	 * @param prevActivations previous activations (output of the previous layer)
	 * @return output of this layer
	 */
	public Tensor3 activations(final Tensor3 prevActivations);
	
	/**
	 * Propagates the error through this layer. This function should change the state of the layer if it needs to:
	 * for example, it may need to update the filters for a layer.
	 * <p>
	 * This function takes as input the derivative of the network's error with respect to the output of this layer,
	 * and returns the derivative of the network's error with respect to the output of the previous layer. This
	 * function may return a tensor with a different shape than the input tensor.
	 * 
	 * @param errorOutputDeriv (partial) derivative of the network's error with respect to the output of this layer
	 * @return (partial) derivative of the network's error with respect to the input to this layer
	 */
	public Tensor3 backprop(final Tensor3 errorOutputDeriv);
}
