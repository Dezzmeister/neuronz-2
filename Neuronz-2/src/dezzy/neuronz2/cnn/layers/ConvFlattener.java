package dezzy.neuronz2.cnn.layers;

import dezzy.neuronz2.arch.layers.Layer;
import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.constructs.Tensor3;
import dezzy.neuronz2.math.constructs.Vector;

/**
 * Defines a way to convert between a tensor from a convolutional neural network
 * to a vector for a fully connected neural network.
 *
 * @author Joe Desmond
 */
public class ConvFlattener implements Layer<Tensor3, Vector> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3882880691102712445L;
	
	/**
	 * Number of expected rows in the input tensor
	 */
	private final int inputRows;
	
	/**
	 * Number of expected columns in the input tensor
	 */
	private final int inputCols;
	
	/**
	 * Number of expected layers in the input tensor
	 */
	private final int inputLayers;
	
	/**
	 * Creates a ConvFlattener expecting tensors of the given shape.
	 * 
	 * @param _inputRows number of expected rows
	 * @param _inputCols number of expected columns
	 * @param _inputLayers number of expected layers
	 */
	public ConvFlattener(final int _inputRows, final int _inputCols, final int _inputLayers) {
		inputRows = _inputRows;
		inputCols = _inputCols;
		inputLayers = _inputLayers;
	}

	/**
	 * Converts the given tensor to a vector. Elements of the tensor are copied into the vector row-wise, then
	 * layer-wise.
	 * <p>
	 * <b>NOTE:</b> Does not alter the values of the data, only the shape.
	 * 
	 * @param prevActivations input tensor
	 * @return output vector
	 */
	@Override
	public Vector forwardPass(final Tensor3 prevActivations) {
		final double[] out = new double[prevActivations.dimension * prevActivations.getLayer(0).rows * prevActivations.getLayer(0).cols];
		
		for (int l = 0; l < prevActivations.dimension; l++) {
			final Matrix layer = prevActivations.getLayer(l);
			layer.copyTo(out, l * layer.rows * layer.cols);
		}
		
		return new Vector(out);
	}
	
	/**
	 * Converts the given vector to a tensor of the expected size, undoing {@link #forwardPass(Tensor3)}.
	 * 
	 * @param errorOutputDeriv input vector
	 * @param isFirstLayer unused
	 * @return output tensor
	 */
	@Override
	public Tensor3 backprop(final Vector errorOutputDeriv, final boolean isFirstLayer) {
		final Matrix[] out = new Matrix[inputLayers];
		
		for (int l = 0; l < inputLayers; l++) {
			final Vector[] vectors = new Vector[inputRows];
			
			for (int row = 0; row < inputRows; row++) {
				final double[] elements = new double[inputCols];
				final int srcIndex = (l * inputRows * inputCols) + (row * inputCols);
				
				errorOutputDeriv.copyTo(srcIndex, elements, 0, inputCols);
				vectors[row] = new Vector(elements);
			}
			
			out[l] = new Matrix(vectors);
		}
		
		return new Tensor3(out);
	}
	
	/**
	 * This layer does not contain any gradients to update, so this function is unused.
	 * 
	 * @param learningRate unused
	 */
	@Override
	public void update(final double learningRate) {
		// TODO Auto-generated method stub
		
	}
	
}
