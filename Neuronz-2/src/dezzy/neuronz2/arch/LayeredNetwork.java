package dezzy.neuronz2.arch;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

import dezzy.neuronz2.arch.error.CompleteErrorFunc;
import dezzy.neuronz2.arch.layers.Layer;
import dezzy.neuronz2.math.constructs.ElementContainer;

/**
 * Part of the new neural network architecture; a network composed of one or more layers.
 * The {@link #network} field may internally contain several layers, but it is treated as 
 * one layer because it is assumed that calling the specified functions in {@link Layer}
 * on {@link #network} will work as expected, regardless of implementation.
 *
 * @author Joe Desmond
 * @param <I> input tensor type to the network (vector, matrix, etc.)
 * @param <O> output tensor type from the network (vector, matrix, etc.)
 */
public class LayeredNetwork<I extends ElementContainer<I>, O extends ElementContainer<O>> implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 3052043222705656121L;

	/**
	 * The actual network; can be either one layer or a 
	 * {@linkplain dezzy.neuronz2.arch.layers.LayerSequence sequence of layers}
	 */
	private final Layer<I, O> network;
	
	/**
	 * The error/cost function and its derivative with respect to the network output
	 */
	private final CompleteErrorFunc<O> errorFunc;
	
	/**
	 * Constructs a network with the given layer(s) and cost function.
	 * 
	 * @param _network layer(s)
	 * @param _errorFunc cost function
	 */
	public LayeredNetwork(final Layer<I, O> _network, final CompleteErrorFunc<O> _errorFunc) {
		network = _network;
		errorFunc = _errorFunc;
	}
	
	/**
	 * Propagates the input through the network and calculates the error given the expected output.
	 * This function may change the state of the {@linkplain #network network}; particularly if any of the
	 * layers have weights.
	 *  
	 * @param input input to the network
	 * @param expectedOutput expected output of the network
	 * @return actual output and error given expected output
	 */
	public final ForwardPassResult<O> forwardPass(final I input, final O expectedOutput) {
		final O actualOutput = network.forwardPass(input);
		final double error = errorFunc.errorFunction.condense(expectedOutput, actualOutput);
		
		return new ForwardPassResult<O>(actualOutput, error);
	}
	
	/**
	 * Propagates the error backwards through the network. Does not update weights; although internal
	 * weight deltas may be updated.
	 * <p>
	 * <b>NOTE:</b> The arguments to this function should be those obtained from the most recent call
	 * to {@link #forwardPass(ElementContainer, ElementContainer) forwardPass()}.
	 * 
	 * @param expectedOutput expected output of the previous forward pass
	 * @param actualOutput actual output of the previous forward pass
	 * @param error error of the previous forward pass 
	 */
	public final void backprop(final O expectedOutput, final O actualOutput, final double error) {
		final O errorDeriv = errorFunc.errorFunctionDerivative.calculate(expectedOutput, actualOutput, error);
		network.backprop(errorDeriv, true);
	}
	
	/**
	 * Updates the learnable parameters in the network.
	 * 
	 * @param learningRate learning rate
	 */
	public final void update(final double learningRate) {
		network.update(learningRate);
	}
	
	/**
	 * Saves this network to a file so that it can be run/trained later (with {@link #loadFrom}).
	 * 
	 * @param path path to file (will be created if it doesn't exist)
	 * @throws IOException if there is a problem creating the {@link FileOutputStream} or {@link ObjectOutputStream}
	 */
	public final void saveAs(final String path) throws IOException {
		final FileOutputStream fos = new FileOutputStream(path);
		final ObjectOutputStream oos = new ObjectOutputStream(fos);
		
		oos.writeObject(this);
		oos.close();
	}
	
	/**
	 * Loads a layered network from a file. Networks can be saved to a file with {@link #saveAs}.
	 * 
	 * @param path path to network file
	 * @return a layered network loaded from <code>path</code>
	 * @throws IOException if there is a problem creating the {@link FileInputStream} or {@link ObjectInputStream}
	 * @throws ClassNotFoundException if the file at <code>path</code> does not contain a {@link LayeredNetwork}
	 */
	public static final LayeredNetwork<?,?> loadFrom(final String path) throws IOException, ClassNotFoundException {
		final FileInputStream fis = new FileInputStream(path);
		final ObjectInputStream ois = new ObjectInputStream(fis);
		final LayeredNetwork<?,?> network = (LayeredNetwork<?,?>) ois.readObject();
		
		ois.close();
		return network;
	}
}
