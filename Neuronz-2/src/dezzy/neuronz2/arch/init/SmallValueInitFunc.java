package dezzy.neuronz2.arch.init;

import java.util.Random;

import dezzy.neuronz2.math.constructs.ElementContainer;
import dezzy.neuronz2.math.constructs.shape.Shape;
import dezzy.neuronz2.math.utility.IndexedGenerator;

/**
 * Initializes parameters to a small value between 0.05 (inclusive) and 0.25 (exclusive).
 *
 * @author Joe Desmond
 */
public class SmallValueInitFunc implements WeightInitFunc {

	@Override
	public <T extends ElementContainer<T>> T initialize(final Random random, final Shape<T> shape, final int numInputs, final int numOutputs, final int numWeights) {
		final IndexedGenerator generator = i -> ((random.nextDouble() * 0.2) + 0.05);
		
		return shape.generate(generator);
	}
	
}
