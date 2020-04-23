package dezzy.neuronz2.cnn.layers;

import java.io.Serializable;

import dezzy.neuronz2.math.constructs.Tensor3;

/**
 * A single layer in a convolutional neural network.
 *
 * @author Joe Desmond
 */
public interface Layer extends Serializable {
	
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
	 * for example, it may need to update a weight delta buffer for the layer. For updating the weights themselves,
	 * {@link #update(double)} should be used.
	 * <p>
	 * <b>IMPORTANT:</b> This function should <b>NOT</b> update the weights for this layer!
	 * <p>
	 * This function takes as input the derivative of the network's error with respect to the output of this layer,
	 * and returns the derivative of the network's error with respect to the output of the previous layer. This
	 * function may return a tensor with a different shape than the input tensor.
	 * <p>
	 * The <code>isFirstLayer</code> flag implements a performance optimization by skipping gradient calculation for the first layer
	 * in a network. This flag does not have to be set to true when calling this function on the first layer;
	 * but if it is, then this function should skip calculating the gradients for the next layer and return
	 * either <code>null</code> or <code>errorOutputDeriv</code>. For example; if this is a convolutional layer and
	 * is the first layer of a network, the return value of this function will not be used because there are no previous
	 * layers to propagate the error to. In a convolutional layer the output would have to be calculated with potentially
	 * several convolutions, so skipping this unnecessary computation can save significant time when training networks.
	 * 
	 * @param errorOutputDeriv (partial) derivative of the network's error with respect to the output of this layer
	 * @param isFirstLayer true if this layer is the first in a network. If this is true, this function should not bother
	 * 			calculating the gradients for the next layer and should instead return null or the gradients that were
	 * 			passed in (<code>errorOutputDeriv</code>)
	 * @return (partial) derivative of the network's error with respect to the input to this layer
	 */
	public Tensor3 backprop(final Tensor3 errorOutputDeriv, final boolean isFirstLayer);
	
	/**
	 * Updates the weights of this layer, if there are any. Some layers (such as {@link PoolingLayer}) will not
	 * use this function.
	 * 
	 * @param learningRate learning rate used for gradient descent; not all layers need this
	 */
	public void update(final double learningRate);
}
