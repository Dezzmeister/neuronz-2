package dezzy.neuronz2.arch;

import java.util.List;
import java.util.Map;

import dezzy.neuronz2.arch.layers.Layer;
import dezzy.neuronz2.math.constructs.ElementContainer;

/**
 * The result of one backward pass through a {@link ParallelLayer}.
 *
 * @author Joe Desmond
 * @param <I> input type
 */
public class ParallelBackwardPass<I extends ElementContainer<I>> {
	
	/**
	 * Partial derivative of this layer's output with respect to its input, or undefined if this is the first layer
	 */
	public final I errorInputDeriv;
	
	/**
	 * Maps every layer (sub-layer) to its weight gradients (there may be several: for example, weights and biases may be separate)
	 */
	public final Map<Layer<?,?>, List<ElementContainer<?>>> gradients;
	
	/**
	 * Constructs a ParallelBackwardPass with the given derivative and gradient mappings.
	 * <p>
	 * <b>Use {@link Map#of()} if creating this object with no gradients.</b>
	 * 
	 * @param _errorInputDeriv derivative of this layer's output with respect to its input
	 * @param _gradients gradients of each sub-layer
	 */
	public ParallelBackwardPass(final I _errorInputDeriv, final Map<Layer<?, ?>, List<ElementContainer<?>>> _gradients) {
		errorInputDeriv = _errorInputDeriv;
		gradients = _gradients;
	}
}
