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
			{0.50f, 0.55f, 0.6f}
		});
		
		final Tensor3 tensor0 = new Tensor3(layer0, layer1);
		System.out.println("Network tensor:\n" + tensor0 + "\n");
		
		final Vector activation0 = new Vector(0.05f, 0.1f, 1);
		System.out.println("First activation vector: " + activation0);
		
		final Vector activation1 = NetworkFunctions.computeOutputVector(layer0, activation0, Functions::sigmoid).append(1); //Append 1 (for biases)
		System.out.println("Second activation vector: " + activation1);
		
		final Vector activation2 = NetworkFunctions.computeOutputVector(layer1, activation1, Functions::sigmoid);
		System.out.println("Third activation vector: " + activation2);
		
		final Vector ideal0 = new Vector(0.01f, 0.99f);
		System.out.println("Ideal output: " + ideal0);
		
		final float totalError0 = NetworkFunctions.computeTotalMSE(ideal0, activation2);
		System.out.println("Total error: " + totalError0);
		
		final Vector errorOutputDeriv = activation2.elementOperation(ideal0, (actual, ideal) -> - (ideal - actual));
		System.out.println("Partial derivative of error wrt output: " + errorOutputDeriv);
		
		final Vector sigmoidDeriv0 = activation2.transform(out -> out * (1 - out));
		System.out.println("Partial derivative of output wrt input (layer 0): " + sigmoidDeriv0);
		
		final Matrix weightDeltas0 = errorOutputDeriv.hadamard(sigmoidDeriv0).outerProduct(activation1);
		System.out.println("Weight gradients (layer 0): \n" + weightDeltas0);
		
		final float eta0 = 0.5f;
		System.out.println("Learning rate: " + eta0);
		
		final Matrix newWeights0 = layer1.minus(weightDeltas0.transform(w -> w * eta0));
		System.out.println("New weights (layer 0): \n" + newWeights0);
		
		System.out.println("Partial derivative of error wrt inputs: " + errorOutputDeriv.hadamard(sigmoidDeriv0));
		System.out.println();
		System.out.println("ALTERNATE");
		System.out.println("Partial derivative of error wrt output: " + errorOutputDeriv);
		
		
		//System.out.println();
	}
}
