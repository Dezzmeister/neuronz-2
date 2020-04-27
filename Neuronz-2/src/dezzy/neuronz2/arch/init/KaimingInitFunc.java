package dezzy.neuronz2.arch.init;

import java.util.Random;

import dezzy.neuronz2.math.constructs.ElementContainer;
import dezzy.neuronz2.math.constructs.shape.Shape;
import dezzy.neuronz2.math.utility.IndexedGenerator;

/**
 * Implements Kaiming-He initialization
 *
 * @author Joe Desmond
 */
final class KaimingInitFunc implements WeightInitFunc {

	@Override
	public <T extends ElementContainer<T>> T initialize(final Random random, final Shape<T> shape, final int numInputs, final int numOutputs, final int numWeights) {		
		final double modifier = Math.sqrt(2 / (numInputs));
		final IndexedGenerator generator = i -> random.nextGaussian() * modifier;
		
		return shape.generate(generator);
	}	
}
