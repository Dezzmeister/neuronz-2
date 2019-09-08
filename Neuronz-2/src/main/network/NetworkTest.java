package main.network;

import main.math.constructs.Matrix;
import main.math.constructs.Tensor3;
import main.math.constructs.Vector;
import main.math.utility.Functions;

/**
 * Contains functions for testing neural networks.
 *
 * @author Joe Desmond
 */
public final class NetworkTest {
	
	public static final void main(final String[] args) {
		final Matrix layer0 = new Matrix(new float[][] {
			{0.15f, 0.2f, 0.35f},
			{0.25f, 0.3f, 0.35f}
		});
		
		final Matrix layer1 = new Matrix(new float[][] {
			{0.4f, 0.45f, 0.6f},
			{0.5f, 0.55f, 0.6f}
		});
		
		final Tensor3 tensor0 = new Tensor3(layer0, layer1);
		System.out.println("Weight tensor:\n" + tensor0 + "\n");
		
		final Vector activation0 = new Vector(0.05f, 0.1f, 1);
		System.out.println("First activation vector: " + activation0);
		
		final Vector activation1 = NetworkFunctions.computeOutputVector(layer0, activation0, Functions::sigmoid).append(1);
		System.out.println("Second activation vector: " + activation1);
		
		final Vector activation2 = NetworkFunctions.computeOutputVector(layer1, activation1, Functions::sigmoid);
		System.out.println("Third activation vector: " + activation2);
	}
}
