package dezzy.neuronz2.ann.error;

import dezzy.neuronz2.arch.error.CompleteErrorFunc;
import dezzy.neuronz2.math.constructs.Vector;

/**
 * Contains various error functions evaluated over vector network outputs.
 *
 * @author Joe Desmond
 */
public class VectorErrorFunctions {
	
	/**
	 * The cross entropy function and its derivative
	 */
	public static final CompleteErrorFunc<Vector> CROSS_ENTROPY = new CrossEntropy();
	
	/**
	 * The mean square error function and its derivative
	 */
	public static final CompleteErrorFunc<Vector> MEAN_SQUARE_ERROR = new MeanSquareError();
}
