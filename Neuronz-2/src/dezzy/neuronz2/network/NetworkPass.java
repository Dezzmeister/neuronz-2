package dezzy.neuronz2.network;

import java.util.concurrent.Callable;

import dezzy.neuronz2.math.constructs.Tensor3;
import dezzy.neuronz2.math.constructs.Vector;


/**
 * One forward and backward pass through the network.
 *
 * @author Joe Desmond
 */
public final class NetworkPass implements Callable<Tensor3> {
	private final Network network;
	private final Vector input;
	private final Vector ideal;
	
	/**
	 * The input vector will be used to calculate activations in the given network, and backpropagation will be performed using <code>ideal</code>
	 * to calculate the gradient.
	 * 
	 * @param _network network to run input vector through
	 * @param _input input vector
	 * @param _ideal ideal output vector
	 */
	public NetworkPass(final Network _network, final Vector _input, final Vector _ideal) {
		network = _network;
		input = _input;
		ideal = _ideal;
	}
	
	/**
	 * Returns the weight gradient after one forward pass and one backward pass through the network.
	 */
	@Override
	public Tensor3 call() throws Exception {
		return network.backprop(input, ideal).weightDeltas;
	}
	
}
