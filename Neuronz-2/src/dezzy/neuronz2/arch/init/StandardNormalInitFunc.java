package dezzy.neuronz2.arch.init;

import java.util.Random;

import dezzy.neuronz2.math.constructs.ElementContainer;
import dezzy.neuronz2.math.constructs.shape.Shape;
import dezzy.neuronz2.math.utility.IndexedGenerator;

/**
 * Initializes a tensor with every value set to a randomly chosen number from a normal
 * distribution with a mean of 0 and a variance of 1.
 *
 * @author Joe Desmond
 */
public class StandardNormalInitFunc implements WeightInitFunc {

	@Override
	public <T extends ElementContainer<T>> T initialize(final Random random, final Shape<T> shape, final int numInputs, final int numOutputs, final int numWeights) {
		final IndexedGenerator generator = i -> random.nextGaussian();
		return shape.generate(generator);
	}
	
}
