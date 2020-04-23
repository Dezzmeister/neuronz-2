package dezzy.neuronz2.arch.test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import dezzy.neuronz2.ann.error.VectorErrorFunctions;
import dezzy.neuronz2.ann.layers.DenseLayer;
import dezzy.neuronz2.arch.ForwardPassResult;
import dezzy.neuronz2.arch.LayeredNetwork;
import dezzy.neuronz2.arch.layers.ElementActivationLayer;
import dezzy.neuronz2.arch.layers.Layer;
import dezzy.neuronz2.arch.layers.LayerSequence;
import dezzy.neuronz2.math.constructs.FuncDerivPair;
import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.constructs.Vector;
import dezzy.neuronz2.math.utility.IndexedGenerator;

/**
 * Some functions to test the new architecture, and ensure that neural networks built with layers
 * from the new architecture can learn correctly.
 *
 * @author Joe Desmond
 */
public class NewNetworkTest {
	
	public static final void main(final String[] args) throws IOException {
		andGateTest();
	}
	
	/**
	 * A test network implementing a logical "and" gate.
	 * @throws IOException 
	 */
	private static final void andGateTest() throws IOException {
		final IndexedGenerator generator = (indices) -> {
			return Math.random() + 0.1;
		};
		
		final Matrix layer0w = Matrix.generate(generator, 4, 2);
		final Matrix layer1w = Matrix.generate(generator, 1, 4);
		
		final Vector layer0b = Vector.generate(generator, 4);
		final Vector layer1b = Vector.generate(generator, 1);
		
		final DenseLayer layer0 = new DenseLayer(layer0w, layer0b);
		final DenseLayer layer1 = new DenseLayer(layer1w, layer1b);
		final ElementActivationLayer<Vector> sigmoid0 = new ElementActivationLayer<>(FuncDerivPair.SIGMOID);
		final ElementActivationLayer<Vector> sigmoid1 = new ElementActivationLayer<>(FuncDerivPair.SIGMOID);
		
		List<Layer<Vector, Vector>> layers = new ArrayList<>();
		layers.add(layer0);
		layers.add(sigmoid0);
		layers.add(layer1);
		layers.add(sigmoid1);
		
		
		final LayerSequence<Vector> sequence = new LayerSequence<Vector>(layers);
		final LayeredNetwork<Vector, Vector> network = new LayeredNetwork<>(sequence, VectorErrorFunctions.MEAN_SQUARE_ERROR);
		
		final Vector input0 = new Vector(0, 0);
		final Vector input1 = new Vector(0, 1);
		final Vector input2 = new Vector(1, 0);
		final Vector input3 = new Vector(1, 1);
		
		final Vector exp0 = new Vector(0);
		final Vector exp1 = new Vector(1);
		
		final Vector[] inputs = {input0, input1, input2, input3};
		final Vector[] expecteds = {exp0, exp0, exp0, exp1};
		
		
		for (int i = 0; i < 10; i++) {
			int successes = 0;
			
			for (int j = 0; j < 100; j++) {
				final int index = (int)(Math.random() * 4);
				final Vector input = inputs[index];
				final Vector expected = expecteds[index];
				
				final ForwardPassResult<Vector> result = network.forwardPass(input, expected);
				
				final int expValue = (int)expected.get(0);
				final Vector actual = result.actualOutput;
				final double actualValue = actual.get(0);
				
				if ((actualValue < 0.5 && expValue == 0) || (actualValue >= 0.5 && expValue == 1)) {
					successes++;
				}
				
				network.backprop(expected, actual, result.error);
				network.update(1);
			}
			
			System.out.println(i + ":\t" + successes + "/100");
		}
		
		//LayeRed Network, or LeaRN
		network.saveAs("networks/new-arch-test/andgate.lrn");
	}
}
