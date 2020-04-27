package dezzy.neuronz2.arch;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import dezzy.neuronz2.arch.layers.Layer;
import dezzy.neuronz2.math.constructs.ElementContainer;

/**
 * A {@linkplain Layer} that supports multithreading. This kind of layer lends itself well to minibatch SGD:
 * several forward/backward passes can be performed on this layer at the same time, and the resulting gradients
 * (returned by {@link #parallelBackprop(ParallelForwardPass, ElementContainer, boolean)}) can be summed
 * and propagated through the layer with {@link #parallelUpdate(ParallelBackwardPass, double)}.
 *
 * @author Joe Desmond
 * @param <I> input tensor type
 * @param <O> output tensor type
 */
public interface ParallelLayer<I extends ElementContainer<I>, O extends ElementContainer<O>> extends Layer<I, O> {
	
	/**
	 * A parallel forward pass through the layer. This function must not modify the state of the layer,
	 * and it should function identically to {@link #forwardPass(ElementContainer)} with one exception:
	 * Instead of saving the latest input to a local variable, this function should instead return a mapping
	 * for it in the {@link ParallelForwardPass}. 
	 * 
	 * @param prevActivations previous activations (output of the previous layer)
	 * @return mappings for the latest input of each layer within this one
	 */
	public ParallelForwardPass<O> parallelForwardPass(final I prevActivations);
	
	/**
	 * Computes backpropagation and returns all the weight deltas. Identical to 
	 * {@link #backprop(ElementContainer, boolean)}, except the gradients are returned and the state of the
	 * layer is not changed. The latest input to this layer is stored in <code>prevForward</code>.
	 * <p>
	 * <b>NOTE:</b> This function should be called only after {@link #parallelForwardPass(ElementContainer)}.
	 * 
	 * @param prevForward result of previous call to {@link #parallelForwardPass(ElementContainer)}
	 * @param errorOutputDeriv (partial) derivative of the network's error with respect to the output of this layer
	 * @param isFirstLayer  true if this layer is the first in a network. If this is true, this function should not bother
	 * 			calculating the gradients for the next layer and should instead return null or the gradients that were
	 * 			passed in (<code>errorOutputDeriv</code>)
	 * @return (partial) derivative of the network's error with respect to the input to this layer
	 */
	public ParallelBackwardPass<I> parallelBackprop(final ParallelForwardPass<O> prevForward, final O errorOutputDeriv, final boolean isFirstLayer);
	
	/**
	 * Propagates gradients back through the layer and updates any sub-layers.
	 * 
	 * @param gradients result of previous backward pass, or a composition of several backward passes
	 * @param learningRate learning rate
	 */
	public void parallelUpdate(final ParallelBackwardPass<?> gradients, final double learningRate);
	
	/**
	 * ParallelLayer-specific version of {@link Layer#loadFrom(String)}:<br>
	 * Saves this ParallelLayer network to a file so that it can be run/trained later.
	 * 
	 * @param path path to file (will be created if it doesn't exist)
	 * @throws IOException if there is a problem creating the {@link FileOutputStream} or {@link ObjectOutputStream}
	 */
	public static void saveAs(final ParallelLayer<?, ?> layer, final String path) throws IOException {
		final FileOutputStream fos = new FileOutputStream(path);
		final ObjectOutputStream oos = new ObjectOutputStream(fos);
		
		oos.writeObject(layer);
		oos.close();
	}
	
	/**
	 * ParallelLayer-specific version of {@link Layer#loadFrom(String)}:<br> 
	 * Loads a parallel layer from a file. Layers can be saved to a file by {@link Layer#saveAs}.
	 * 
	 * @param path path to layer network file
	 * @return a layer loaded from <code>path</code>
	 * @throws IOException if there is a problem creating the {@link FileInputStream} or {@link ObjectInputStream}
	 * @throws ClassNotFoundException if the file at <code>path</code> does not contain a {@link Layer}
	 */
	@SuppressWarnings("unchecked")
	public static <I extends ElementContainer<I>, O extends ElementContainer<O>> ParallelLayer<I, O> loadFrom(final String path) throws IOException, ClassNotFoundException {
		final FileInputStream fis = new FileInputStream(path);
		final ObjectInputStream ois = new ObjectInputStream(fis);
		final ParallelLayer<I, O> layer = (ParallelLayer<I, O>) ois.readObject();
		
		ois.close();
		return layer;
	}
}
