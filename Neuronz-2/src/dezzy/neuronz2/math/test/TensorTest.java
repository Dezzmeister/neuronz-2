package dezzy.neuronz2.math.test;

import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.constructs.Tensor3;

/**
 * A collection of tests for {@link #Tensor}.
 *
 * @author Joe Desmond
 */
public class TensorTest {
	
	public static final void main(final String[] args) {
		convTest();
	}
	
	private static final void convTest() {
		final Matrix layer0 = new Matrix(new double[][] {
			{1, 2, 4, 1},
			{6, 3, 0.4, 4},
			{7, 1, 2, 1},
			{0.2, 0.4, 0.6, 0.8}
		});
		
		final Matrix layer1 = new Matrix(new double[][] {
			{-0.3, 0.2, -0.1, 3},
			{0.4, 0.5, 0.5, 0.15},
			{1, 2, 3, 4},
			{0.5, 0.3, 1.5, 2.2}
		});
		
		final Matrix layer2 = new Matrix(new double[][] {
			{0.9, 1.5, 0.4, 2.2},
			{1.4, 0.7, 0.4, 1.99},
			{0.8, 0.5, 0.75, 0.25},
			{-0.25, -1, 2, 1}
		});
		
		final Tensor3 tensor = new Tensor3(layer0, layer1, layer2);
		
		final Matrix kern0 = new Matrix(new double[][] {
			{0, 1},
			{-1, 0}
		});
		
		final Matrix kern1 = new Matrix(new double[][] {
			{0.5, -0.25},
			{0.75, -0.15}
		});
		
		final Matrix kern2 = new Matrix(new double[][] {
			{1, 2},
			{3, 4}
		});
		
		final Tensor3 kernel = new Tensor3(kern0, kern1, kern2);
		
		final Tensor3 result = tensor.convolve(kernel, 1, d -> d);
		
		System.out.println("Tensor:\n" + tensor);
		System.out.println("\nKernel:\n" + kernel);
		System.out.println("\nResult:\n" + result);
		//The first element of the result should be 6.925
	}
}
