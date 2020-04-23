package dezzy.neuronz2.ann.error;

import dezzy.neuronz2.arch.error.CompleteErrorFunc;
import dezzy.neuronz2.arch.error.TensorErrorFunc;
import dezzy.neuronz2.arch.error.TensorErrorFuncDeriv;
import dezzy.neuronz2.math.constructs.Vector;

/**
 * The cross entropy error function and its derivative with respect to the network output.
 *
 * @author Joe Desmond
 */
public class CrossEntropy extends CompleteErrorFunc<Vector> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1757627346022852092L;
	
	/**
	 * Cross entropy function
	 */
	private static final TensorErrorFunc<Vector> crossEntropy = (expected, actual) -> {
		final Vector logActual = actual.transform(Math::log);
		final double dotProduct = expected.innerProduct(logActual);
		
		return -dotProduct;
	};
	
	/**
	 * Cross entropy derivative
	 */
	private static final TensorErrorFuncDeriv<Vector> crossEntropyDeriv = (expected, actual, error) -> {
		final Vector out = expected.elementOperation(actual, (e, a) -> -e/a);
		
		return out;
	};

	/**
	 * Constructs a {@link CompleteErrorFunc} for vectors using the cross entropy cost function.
	 */
	public CrossEntropy() {
		super(crossEntropy, crossEntropyDeriv);
	}	
}
