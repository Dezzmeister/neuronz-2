package dezzy.neuronz2.arch.init;

import java.util.Random;

import dezzy.neuronz2.math.constructs.ElementContainer;
import dezzy.neuronz2.math.constructs.shape.Shape;
import dezzy.neuronz2.math.utility.IndexedGenerator;

/**
 * Implements Xavier's initialization
 *
 * @author Joe Desmond
 */
final class XavierInitFunc implements WeightInitFunc {

	@Override
	public <T extends ElementContainer<T>> T initialize(final Random random, final Shape<T> shape, final int numInputs, final int numOutputs, final int numWeights) {
		final double modifier = Math.sqrt(6 / ((double)(numInputs + numOutputs)));
		final IndexedGenerator generator = i -> ((random.nextDouble() * 2) - 1) * modifier;
		
		return shape.generate(generator);
	}
	
}
