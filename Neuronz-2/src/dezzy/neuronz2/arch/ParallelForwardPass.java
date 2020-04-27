package dezzy.neuronz2.arch;

import java.util.Map;

import dezzy.neuronz2.arch.layers.Layer;
import dezzy.neuronz2.math.constructs.ElementContainer;

/**
 * Contains the result of one forward pass through a layer. This is used by
 * {@link ParallelLayer} to parallelize training.
 *
 * @author Joe Desmond
 */
public class ParallelForwardPass<O extends ElementContainer<O>> {
	
	/**
	 * The output of the layer
	 */
	public final O output;
	
	/**
	 * Maps each layer to the latest input to that layer
	 */
	public final Map<Layer<?, ?>, ElementContainer<?>> latestInputs;
	
	/**
	 * Maps each layer to the latest output from that layer
	 */
	public final Map<Layer<?, ?>, ElementContainer<?>> latestOutputs;
	
	/**
	 * Creates a ParallelForwardPass with the given output, latest input mappings,
	 * and latest output mappings (optional, but some layers such as
	 * {@link dezzy.neuronz2.ann.layers.SoftmaxLayer SoftmaxLayer} require it).
	 * <p>
	 * <b>If creating an object where one of the two Maps will not be needed, use
	 * {@link Map#of()}.</b>
	 * 
	 * @param _output output of the layer
	 * @param _latestInputs latest inputs to each sub-layer
	 * @param _latestOutputs latest outputs from each sub-layer
	 */
	public ParallelForwardPass(final O _output, final Map<Layer<?, ?>, ElementContainer<?>> _latestInputs, final Map<Layer<?, ?>, ElementContainer<?>> _latestOutputs) {
		output = _output;
		latestInputs = _latestInputs;
		latestOutputs = _latestOutputs;
	}
}
