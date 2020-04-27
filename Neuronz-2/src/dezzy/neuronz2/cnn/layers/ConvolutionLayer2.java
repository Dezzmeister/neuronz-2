package dezzy.neuronz2.cnn.layers;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import dezzy.neuronz2.arch.ParallelBackwardPass;
import dezzy.neuronz2.arch.ParallelForwardPass;
import dezzy.neuronz2.arch.ParallelLayer;
import dezzy.neuronz2.arch.init.WeightInitFunc;
import dezzy.neuronz2.arch.layers.Layer;
import dezzy.neuronz2.math.constructs.ElementContainer;
import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.constructs.Tensor3;
import dezzy.neuronz2.math.constructs.Tensor4;
import dezzy.neuronz2.math.constructs.Vector;
import dezzy.neuronz2.math.constructs.shape.Tensor4Shape;
import dezzy.neuronz2.math.constructs.shape.VectorShape;
import dezzy.neuronz2.math.utility.DoubleApplier;

/**
 * A convolutional layer in a convolutional neural network. Uses 3D filters 
 * ({@linkplain Tensor3 rank 3 tensors}); works on input with 1 or more channels. 
 * Multiple filters can be used, so the filters are represented as a 
 * {@linkplain Tensor4 rank 4 tensor}.
 *
 * @author Joe Desmond
 */
public class ConvolutionLayer2 implements ParallelLayer<Tensor3, Tensor3> {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 734117236105763152L;
	
	/**
	 * The filters in this layer. There can be more than 1 filter, and the filters themselves
	 * are 3D tensors
	 */
	private Tensor4 filters;
	
	/**
	 * The biases in this layer. There is one bias per filter
	 */
	private Vector biases;
	
	/**
	 * The accumulated filter gradients from calls to {@link #backprop(Tensor3, boolean)}
	 */
	private Tensor4 filterDeltas;
	
	/**
	 * The accumulated bias gradients from calls to {@link #backprop(Tensor3, boolean)}
	 */
	private Vector biasDeltas;
	
	/**
	 * The latest input to this layer (latest input to {@link #forwardPass(Tensor3)},
	 * used in backpropagation
	 */
	private Tensor3 latestInput;
	
	/**
	 * Defined explicitly and used in convolutions, because {@link Matrix#convolve(Matrix, int, DoubleApplier)}
	 * requires a functional parameter
	 */
	private static final DoubleApplier NON_MODIFIER = d -> d;
	
	/**
	 * Constructs a convolutional layer with the given initial filters and biases.
	 * 
	 * @param _filters filters
	 * @param _biases biases
	 */
	public ConvolutionLayer2(final Tensor4 _filters, final Vector _biases) {
		filters = _filters;
		biases = _biases;
	}
	
	/**
	 * Generates a convolutional layer with the given hyperparameters and initializes the weights and biases using the given functions.
	 * 
	 * @param random random number generator
	 * @param weightInitializer weight initialization function
	 * @param biasInitializer bias initialization function
	 * @param numFilters number of filters in the convolutional layer
	 * @param filterLayers number of layers (matrices) in each filter
	 * @param filterRows number of rows in each layer
	 * @param filterCols number of columns in each layer
	 * @return a new convolutional layer with weights and biases initialized
	 */
	public static ConvolutionLayer2 generate(final Random random, final WeightInitFunc weightInitializer, final WeightInitFunc biasInitializer, final int numFilters, final int filterLayers, final int filterRows, final int filterCols) {
		final Tensor4Shape weightShape = new Tensor4Shape(numFilters, filterLayers, filterRows, filterCols);
		final VectorShape biasShape = new VectorShape(numFilters);
		
		final int numInputs = filterRows * filterCols * filterLayers;
		final int numOutputs = numInputs;
		final int numWeights = filterRows * filterCols * filterLayers * numFilters;
		
		final Tensor4 weights = weightInitializer.initialize(random, weightShape, numInputs, numOutputs, numWeights);
		final Vector biases = biasInitializer.initialize(random, biasShape, numInputs, numOutputs, numWeights);
		
		return new ConvolutionLayer2(weights, biases);
	}
	
	/**
	 * Convolves the given tensor with each filter (in {@link #filters}) and takes the first layer
	 * (first matrix) in the result, then adds the appropriate bias to every element. 
	 * These matrices are layered together to form a tensor, which is returned.
	 * Each convolution should only return a one-layer result, because this layer only works
	 * when the input tensor has the same depth as the filter tensors.
	 * <p>
	 * The input (<code>prevActivations</code>) is saved internally to be used in backpropagation.
	 * 
	 * @param prevActivations input to this layer
	 * @return output of this layer
	 */
	@Override
	public Tensor3 forwardPass(final Tensor3 prevActivations) {
		latestInput = prevActivations;
		
		final Matrix[] output = new Matrix[filters.dimension];
		
		for (int i = 0; i < filters.dimension; i++) {
			final Tensor3 kernel = filters.getTensor(i);
			final Tensor3 convolved = prevActivations.convolve(kernel, 1, NON_MODIFIER);
			final Matrix result = convolved.getLayer(0);
			final double bias = biases.get(i);
			
			output[i] = result.transform(d -> d + bias);
		}
		
		return new Tensor3(output);
	}
	
	/**
	 * Performs backpropagation on this layer, in several steps:
	 * <ol>
	 * <li>Computes the filter gradients. This gradient for a channel layer in a filter tensor is given by
	 * convolving the corresonding channel layer from the latest input matrix with the filter's
	 * layer in <code>errorOutputDeriv</code>.</li>
	 * <li>Computes the bias gradients. There is one bias for every filter, and the bias gradient for a
	 * filter is simply the sum of every element in the filter's layer in <code>errorOutputDeriv</code>.</li>
	 * <li>Updates the internal filter and matrix gradients. These gradients are not propagated to the
	 * actual filters and biases until {@link #update(double)} is called. Until then, a sum of the gradients
	 * is kept.</li>
	 * <li>Checks if this is the first layer in the network (if <code>isFirstLayer</code> is set to true)
	 * and returns immediately, since the remainder of the function calculates the derivative of the error
	 * with respect to this layer's input. This derivative is not needed if this is the first layer;
	 * otherwise, it should be propagated to the previous layers.</li>
	 * <li>Pads the layers in <code>errorOutputDeriv</code>, so that a full convolution can be performed and the result
	 * of this convolution will have the same size as the input tensor.</li>
	 * <li>Computes the derivative of the error with respect to this layer's input. For each channel, every layer of 
	 * the padded <code>errorOutputDeriv</code> is convolved with the channel layer (rotated 180 degrees) in it's corresponding filter.
	 * The derivative of the error with respect to the layer's input for the channel is given by summing
	 * the results of these convolutions.</li>
	 * </ol>
	 * 
	 * @param errorOutputDeriv (partial) derivative of the network error with respect to this layer's output
	 * @param isFirstLayer true if this is the first layer in the network. Unlike some other layers,
	 * 			({@link PoolingLayer}, for instance) this parameter is used
	 * @return (partial) derivative of the network error with respect to this layer's input
	 */
	@Override
	public Tensor3 backprop(final Tensor3 errorOutputDeriv, final boolean isFirstLayer) {
		final Tensor3[] newFilterDeltas = new Tensor3[filters.dimension];
		final double[] newBiasDeltas = new double[filters.dimension];
		
		// Calculate filter and bias gradients
		for (int i = 0; i < errorOutputDeriv.dimension; i++) {
			final Matrix derivMatrix = errorOutputDeriv.getLayer(i);
			final Matrix[] channelGradients = new Matrix[latestInput.dimension];
			
			for (int m = 0; m < latestInput.dimension; m++) {
				final Matrix channel = latestInput.getLayer(m);
				
				channelGradients[m] = channel.convolve(derivMatrix, 1, NON_MODIFIER);
			}
			
			newFilterDeltas[i] = new Tensor3(channelGradients);
			
			newBiasDeltas[i] = derivMatrix.sum();
		}
		
		// Update filter deltas
		final Tensor4 filterGradient = new Tensor4(newFilterDeltas);
		
		if (filterDeltas == null) {
			filterDeltas = filterGradient;
		} else {
			filterDeltas = filterDeltas.plus(filterGradient);
		}
		
		// Update bias deltas
		final Vector biasGradient = new Vector(newBiasDeltas);
		
		if (biasDeltas == null) {
			biasDeltas = biasGradient;
		} else {
			biasDeltas = biasDeltas.plus(biasGradient);
		}
		
		if (isFirstLayer) {
			return null;
		}
		
		
		// Create padded versions of the derivative matrices to compute the next derivatives with
		final Matrix[] paddedDerivatives = new Matrix[errorOutputDeriv.dimension];
		
		for (int i = 0; i < paddedDerivatives.length; i++) {
			final Matrix derivative = errorOutputDeriv.getLayer(i);
			final Matrix filterLayer = filters.getTensor(i).getLayer(0);
			final int padRows = filterLayer.rows - 1;
			final int padCols = filterLayer.cols - 1;
			
			final Matrix padded = derivative.padZero(padRows, padCols);
			paddedDerivatives[i] = padded;
		}
		
		// Compute the derivative of the error with respect to this layer's input
		final Matrix[] output = new Matrix[latestInput.dimension];		
		
		for (int channel = 0; channel < output.length; channel++) {
			Matrix deltas = null;
			
			for (int filterIndex = 0; filterIndex < filters.dimension; filterIndex++) {
				final Matrix derivative = paddedDerivatives[filterIndex];
				final Matrix filter = filters.getTensor(filterIndex).getLayer(channel).rotate180();
				
				final Matrix convolved = derivative.convolve(filter, 1, NON_MODIFIER);
				
				if (deltas == null) {
					deltas = convolved;
				} else {
					deltas = deltas.plus(convolved);
				}
			}
			
			output[channel] = deltas;
		}
		
		return new Tensor3(output);
	}
	
	/**
	 * Multiplies the accumulated filter and bias gradients by the learning rate, and uses
	 * gradient descent to update this layer's filters and biases.
	 * 
	 * @param learningRate the learning rate
	 */
	@Override
	public void update(final double learningRate) {
		final Tensor4 scaledFilterGradient = filterDeltas.transform(w -> learningRate * w);
		final Vector scaledBiasGradient = biasDeltas.transform(w -> learningRate * w);
		
		filters = filters.minus(scaledFilterGradient);
		biases = biases.minus(scaledBiasGradient);
		
		filterDeltas = null;
		biasDeltas = null;
	}
	
	/**
	 * Returns the number of bias units plus the number of units in the filter tensor.
	 * 
	 * @return the total number of learnable parameters in this layer
	 */
	@Override
	public int parameterCount() {
		final Tensor3 t = filters.getTensor(0);
		final Matrix m = t.getLayer(0);
		return biases.dimension + (filters.dimension * t.dimension * m.rows * m.cols);
	}
	
	/**
	 * Returns one because this layer is not composed of any sublayers.
	 * 
	 * @return one
	 */
	@Override
	public int sublayers() {
		return 1;
	}

	@Override
	public ParallelForwardPass<Tensor3> parallelForwardPass(final Tensor3 prevActivations) {
		final Map<Layer<?, ?>, ElementContainer<?>> latestInputs = new HashMap<>();
		latestInputs.put(this, prevActivations);
		
		final Matrix[] output = new Matrix[filters.dimension];
		
		for (int i = 0; i < filters.dimension; i++) {
			final Tensor3 kernel = filters.getTensor(i);
			final Tensor3 convolved = prevActivations.convolve(kernel, 1, NON_MODIFIER);
			final Matrix result = convolved.getLayer(0);
			final double bias = biases.get(i);
			
			output[i] = result.transform(d -> d + bias);
		}
		
		final Tensor3 nextActivations = new Tensor3(output);
		
		return new ParallelForwardPass<>(nextActivations, latestInputs, Map.of());
	}

	@Override
	public ParallelBackwardPass<Tensor3> parallelBackprop(final ParallelForwardPass<Tensor3> prevForward, final Tensor3 errorOutputDeriv, final boolean isFirstLayer) {
		final Map<Layer<?, ?>, List<ElementContainer<?>>> gradients = new HashMap<>();
		
		final Tensor3[] newFilterDeltas = new Tensor3[filters.dimension];
		final double[] newBiasDeltas = new double[filters.dimension];
		
		final Tensor3 prevLatestInput = (Tensor3) prevForward.latestInputs.get(this);
		
		// Calculate filter and bias gradients
		for (int i = 0; i < errorOutputDeriv.dimension; i++) {
			final Matrix derivMatrix = errorOutputDeriv.getLayer(i);
			final Matrix[] channelGradients = new Matrix[prevLatestInput.dimension];
			
			for (int m = 0; m < prevLatestInput.dimension; m++) {
				final Matrix channel = prevLatestInput.getLayer(m);
				
				channelGradients[m] = channel.convolve(derivMatrix, 1, NON_MODIFIER);
			}
			
			newFilterDeltas[i] = new Tensor3(channelGradients);
			
			newBiasDeltas[i] = derivMatrix.sum();
		}
		
		// Update filter deltas
		final Tensor4 filterGradient = new Tensor4(newFilterDeltas);
		
		// Update bias deltas
		final Vector biasGradient = new Vector(newBiasDeltas);
		
		gradients.put(this, List.of(filterGradient, biasGradient));
		
		if (isFirstLayer) {
			return new ParallelBackwardPass<>(null, gradients);
		}
		
		
		// Create padded versions of the derivative matrices to compute the next derivatives with
		final Matrix[] paddedDerivatives = new Matrix[errorOutputDeriv.dimension];
		
		for (int i = 0; i < paddedDerivatives.length; i++) {
			final Matrix derivative = errorOutputDeriv.getLayer(i);
			final Matrix filterLayer = filters.getTensor(i).getLayer(0);
			final int padRows = filterLayer.rows - 1;
			final int padCols = filterLayer.cols - 1;
			
			final Matrix padded = derivative.padZero(padRows, padCols);
			paddedDerivatives[i] = padded;
		}
		
		// Compute the derivative of the error with respect to this layer's input
		final Matrix[] output = new Matrix[prevLatestInput.dimension];		
		
		for (int channel = 0; channel < output.length; channel++) {
			Matrix deltas = null;
			
			for (int filterIndex = 0; filterIndex < filters.dimension; filterIndex++) {
				final Matrix derivative = paddedDerivatives[filterIndex];
				final Matrix filter = filters.getTensor(filterIndex).getLayer(channel).rotate180();
				
				final Matrix convolved = derivative.convolve(filter, 1, NON_MODIFIER);
				
				if (deltas == null) {
					deltas = convolved;
				} else {
					deltas = deltas.plus(convolved);
				}
			}
			
			output[channel] = deltas;
		}
		
		return new ParallelBackwardPass<>(new Tensor3(output), gradients);
	}

	@Override
	public void parallelUpdate(final ParallelBackwardPass<?> gradients, double learningRate) {
		final List<ElementContainer<?>> gradientList = gradients.gradients.get(this);
		
		final Tensor4 prevFilterDeltas = (Tensor4) gradientList.get(0);
		final Vector prevBiasDeltas = (Vector) gradientList.get(1);
		
		final Tensor4 scaledFilterGradient = prevFilterDeltas.transform(w -> learningRate * w);
		final Vector scaledBiasGradient = prevBiasDeltas.transform(w -> learningRate * w);
		
		filters = filters.minus(scaledFilterGradient);
		biases = biases.minus(scaledBiasGradient);
	}
}
