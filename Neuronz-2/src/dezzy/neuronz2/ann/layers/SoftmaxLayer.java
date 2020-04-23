package dezzy.neuronz2.ann.layers;

import dezzy.neuronz2.arch.layers.TensorActivationLayer;
import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.constructs.Vector;

/**
 * A softmax layer in a neural network. Implemented as a separate layer instead of an activation function
 * because the derivative of this function is a matrix, and it needs to be multiplied by the 
 * input derivative vector during backpropagation.
 *
 * @author Joe Desmond
 */
public class SoftmaxLayer extends TensorActivationLayer<Vector> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2524448557764427093L;
	
	/**
	 * The latest output from the layer
	 */
	private Vector latestOutput;
	
	/**
	 * Constructs a softmax layer. There are no weights in this layer; this layer only applies the softmax
	 * function to its input.
	 */
	public SoftmaxLayer() {
		super(null);
	}
	
	/**
	 * Applies the softmax function to the input vector and returns the result. Before returning, saves
	 * the output internally for backpropagation.
	 * 
	 * @param prevActivations input vector (output of previous layer)
	 * @return output vector of this layer
	 */
	@Override
	public Vector forwardPass(final Vector prevActivations) {
		final Vector raised = prevActivations.transform(d -> Math.exp(d));
		final double sum = raised.sum();
		
		latestOutput = raised.transform(d -> d / sum);
		return latestOutput;
	}
	
	/**
	 * Performs backpropagation on this layer. The partial derivative of the softmax output with respect
	 * to every element is taken first, which yields a jacobian matrix. The derivative of the error with respect
	 * to this layer's input is obtained by multiplying this matrix by the given derivative vector.
	 * 
	 * @param errorOutputDeriv (partial) derivative of the network error with respect to this layer's output
	 * @param isFirstLayer unused
	 * @return (partial) derivative of the network error with respect to this layer's input
	 */
	public Vector backprop(final Vector errorOutputDeriv, final boolean isFirstLayer) {
		final double[][] values = new double[latestOutput.dimension][latestOutput.dimension];
		
		for (int row = 0; row < values.length; row++) {
			for (int col = 0; col < values[0].length; col++) {
				if (row == col) {
					values[row][col] = latestOutput.get(row) * (1 - latestOutput.get(row));
				} else {
					values[row][col] = -latestOutput.get(row) * latestOutput.get(col);
				}
			}
		}
		
		final Matrix jacobian = new Matrix(values);
		
		return jacobian.multiply(errorOutputDeriv);
	}
}
