package dezzy.neuronz2.arch.layers;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

import dezzy.neuronz2.cnn.layers.PoolingLayer;
import dezzy.neuronz2.math.constructs.ElementContainer;

/**
 * A single layer in a neural network and the building block of large neural networks.
 *
 * @author Joe Desmond
 * @param <I> The input to this layer (examples: may be a {@linkplain dezzy.neuronz2.math.constructs.Matrix matrix},
 * 			{@linkplain dezzy.neuronz2.math.constructs.Tensor3 rank 3 tensor}, etc.)
 * @param <O> The output of this layer
 */
public interface Layer<I extends ElementContainer<I>, O extends ElementContainer<O>> extends Serializable {
	
	/**
	 * Get the activations for this layer (the result of forward propagation of the previous activations).
	 * This function may return a result with a different shape than the input.
	 * 
	 * @param prevActivations previous activations (output of the previous layer)
	 * @return output of this layer
	 */
	public O forwardPass(final I prevActivations);
	
	/**
	 * Propagates the error through this layer. This function should change the state of the layer if it needs to:
	 * for example, it may need to update a weight delta buffer for the layer. For updating the weights themselves,
	 * {@link #update(double)} should be used.
	 * <p>
	 * <b>IMPORTANT:</b> This function should <b>NOT</b> update the weights for this layer!
	 * <p>
	 * This function takes as input the derivative of the network's error with respect to the output of this layer,
	 * and returns the derivative of the network's error with respect to the output of the previous layer. This
	 * function may return a result with a different shape than the input.
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
	public I backprop(final O errorOutputDeriv, final boolean isFirstLayer);
	
	/**
	 * Updates the weights of this layer, if there are any. Some layers (such as {@link PoolingLayer}) will not
	 * use this function.
	 * 
	 * @param learningRate learning rate used for gradient descent; not all layers need this
	 */
	public void update(final double learningRate);
	
	/**
	 * Returns the total number of learnable parameters in this network.
	 * 
	 * @return number of learnable parameters in the network
	 */
	public int parameterCount();
	
	/**
	 * Returns the total number of sub-layers contained in this layer. Layers can be composed of one or more
	 * "sub-layers" chained together, and this function returns the number of layers that have been
	 * chained together to form this one.
	 * <p>
	 * <b>Note:</b> If a layer does not have any sublayers, then it should count itself as a sublayer, 
	 * so any layer that is not a composition of layers should return a value of 1.
	 * 
	 * @return number of sublayers
	 */
	public int sublayers();
	
	/**
	 * Saves this layer network to a file so that it can be run/trained later.
	 * 
	 * @param path path to file (will be created if it doesn't exist)
	 * @throws IOException if there is a problem creating the {@link FileOutputStream} or {@link ObjectOutputStream}
	 */
	public static void saveAs(final Layer<?, ?> layer, final String path) throws IOException {
		final FileOutputStream fos = new FileOutputStream(path);
		final ObjectOutputStream oos = new ObjectOutputStream(fos);
		
		oos.writeObject(layer);
		oos.close();
	}
	
	/**
	 * Loads a layer from a file. Layers can be saved to a file by {@link Layer#saveAs}.
	 * 
	 * @param path path to layer network file
	 * @return a layer loaded from <code>path</code>
	 * @throws IOException if there is a problem creating the {@link FileInputStream} or {@link ObjectInputStream}
	 * @throws ClassNotFoundException if the file at <code>path</code> does not contain a {@link Layer}
	 */
	@SuppressWarnings("unchecked")
	public static <I extends ElementContainer<I>, O extends ElementContainer<O>> Layer<I, O> loadFrom(final String path) throws IOException, ClassNotFoundException {
		final FileInputStream fis = new FileInputStream(path);
		final ObjectInputStream ois = new ObjectInputStream(fis);
		final Layer<I, O> layer = (Layer<I, O>) ois.readObject();
		
		ois.close();
		return layer;
	}
}
