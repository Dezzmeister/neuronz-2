package dezzy.neuronz2.arch.init;

import java.util.Random;

import dezzy.neuronz2.math.constructs.ElementContainer;
import dezzy.neuronz2.math.constructs.shape.Shape;
import dezzy.neuronz2.math.utility.IndexedGenerator;

/**
 * Initializes weights/biases to zero
 *
 * @author Joe Desmond
 */
public class ZeroInitFunc implements WeightInitFunc {

	@Override
	public <T extends ElementContainer<T>> T initialize(final Random random, final Shape<T> shape, final int numInputs, final int numOutputs, final int numWeights) {
		final IndexedGenerator generator = i -> 0;
		
		return shape.generate(generator);
	}
	
}
