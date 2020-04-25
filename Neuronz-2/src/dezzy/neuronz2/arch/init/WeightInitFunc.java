package dezzy.neuronz2.arch.init;

import java.util.Random;

import dezzy.neuronz2.math.constructs.ElementContainer;
import dezzy.neuronz2.math.constructs.shape.Shape;

/**
 * A function that is used to initialize weights or biases in a layer. Parameters should be initialized randomly,
 * and this function allows different techniques to be implemented to ensure that gradients
 * don't explode/vanish.
 *
 * @author Joe Desmond
 */
@FunctionalInterface
public interface WeightInitFunc {
	
	/**
	 * Kaiming iniialization (useful for ReLU)
	 */
	public static final WeightInitFunc KAIMING_INIT = new KaimingInitFunc();
	
	/**
	 * Xavier initialization (useful for tanh and sigmoid)
	 */
	public static final WeightInitFunc XAVIER_INIT = new XavierInitFunc();
	
	/**
	 * Accepts a desired tensor shape and generates a tensor with this shape. Accepts other hyperparameters,
	 * which can be used to modify the initial weights in some way (example: {@link KaimingInitFunc Kaiming Initialization}).
	 * <p>
	 * <b>NOTE:</b> This function can also be used to generate biases.
	 * 
	 * @param <T> tensor type
	 * @param random random number generator
	 * @param shape shape of the tensor to be generated
	 * @param numInputs number of input units in the layer
	 * @param numOutputs number of output units in the layer
	 * @param numWeights number of weights in the layer
	 * @return a single weight
	 */
	<T extends ElementContainer<T>> T initialize(final Random random, final Shape<T> shape, final int numInputs, final int numOutputs, final int numWeights);
}
