package dezzy.neuronz2.ann.error;

import dezzy.neuronz2.arch.error.CompleteErrorFunc;
import dezzy.neuronz2.arch.error.TensorErrorFunc;
import dezzy.neuronz2.arch.error.TensorErrorFuncDeriv;
import dezzy.neuronz2.math.constructs.Vector;

/**
 * The mean-square-error (MSE) cost function and its derivative.
 *
 * @author Joe Desmond
 */
public class MeanSquareError extends CompleteErrorFunc<Vector> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6649285220199188831L;
	
	/**
	 * The MSE function
	 */
	private static final TensorErrorFunc<Vector> mse = (expected, actual) -> {
		double sum = 0;
		
		for (int i = 0; i < expected.dimension; i++) {
			final double e = expected.get(i);
			final double a = actual.get(i);
			
			sum += 0.5 * (e - a) * (e - a);
		}
		
		return sum;
	};
	
	/**
	 * The derivative of the MSE function with respect to the actual network output
	 */
	private static final TensorErrorFuncDeriv<Vector> mseDeriv = (expected, actual, error) -> {
		return actual.minus(expected);
	};
	
	/**
	 * Constructs a {@link CompleteErrorFunc} for vectors using the mean-square-error cost function.
	 */
	public MeanSquareError() {
		super(mse, mseDeriv);
	}	
}
