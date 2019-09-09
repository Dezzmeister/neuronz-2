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
		
		
		final Network network = new Network(2, 4, 3);
		
		final Matrix layer0 = network.weightTensor.getLayer(0);
		final Matrix layer1 = network.weightTensor.getLayer(1);
		
		/*
		final Matrix layer0 = new Matrix(new float[][] {
			{0.15f, 0.2f, 0.35f},
			{0.25f, 0.3f, 0.35f}
		});
		
		final Matrix layer1 = new Matrix(new float[][] {
			{0.4f, 0.45f, 0.6f},
			{0.50f, 0.55f, 0.6f}
		});
		*/
		
		final Tensor3 tensor0 = new Tensor3(layer0, layer1);
		System.out.println("Network tensor:\n" + tensor0 + "\n");
		
		final Vector activation0 = new Vector(0.05f, 0.1f, 1);
		System.out.println("First activation vector (layer 2): " + activation0);
		
		final Vector activation1 = NetworkFunctions.computeOutputVector(layer0, activation0, Functions::sigmoid).append(1); //Append 1 (for biases)
		System.out.println("Second activation vector (layer 1): " + activation1);
		
		final Vector activation2 = NetworkFunctions.computeOutputVector(layer1, activation1, Functions::sigmoid);
		System.out.println("Third activation vector (layer 0): " + activation2);
		
		final Vector ideal0 = new Vector(0.01f, 0.99f, 0.04f);
		System.out.println("Ideal output: " + ideal0);
		
		final float totalError0 = NetworkFunctions.computeTotalMSE(ideal0, activation2);
		System.out.println("Total error: " + totalError0);
		
		final Vector errorOutputDeriv = activation2.elementOperation(ideal0, (actual, ideal) -> - (ideal - actual));
		System.out.println("Partial derivative of error wrt output (layer 0): " + errorOutputDeriv);
		
		final Vector sigmoidDeriv0 = activation2.transform(out -> out * (1 - out)); //out/net
		System.out.println("Partial derivative of output wrt input (layer 0): " + sigmoidDeriv0);
		
		final Vector errorInputDeriv0 = errorOutputDeriv.hadamard(sigmoidDeriv0);
		
		final Matrix weightDeltas0 = errorInputDeriv0.outerProduct(activation1);
		System.out.println("Weight gradients (layer 0): \n" + weightDeltas0);
		
		final float eta0 = 0.5f;
		System.out.println("Learning rate: " + eta0);
		
		final Matrix newWeights0 = layer1.minus(weightDeltas0.transform(w -> w * eta0));
		System.out.println("New weights (layer 0): \n" + newWeights0);
		System.out.println();
		
		final Vector activationDeriv1 = activation1.transform(out -> out * (1 - out));
		System.out.println("Partial derivative of layer 1 outputs wrt layer 1 inputs: " + activationDeriv1);
		
		final Vector layerDeriv0 = layer0.transpose().multiply(activationDeriv1.removeLastElement());
		System.out.println("Partial derivative of layer 0 outputs wrt layer 1 outputs: " + layerDeriv0);
		
		final Vector nextError0 = layer1.transpose().multiply(errorInputDeriv0);
		System.out.println("Partial derivative of error wrt output (layer 1): " + nextError0);
		
		final Vector sigmoidDeriv1 = activation1.transform(out -> out * (1 - out));
		System.out.println("Partial derivative of output wrt input (layer 1): " + sigmoidDeriv1);
		
		final Vector errorInputDeriv1 = nextError0.hadamard(sigmoidDeriv1);
		System.out.println("Partial derivative of error wrt input (layer 1): " + errorInputDeriv1);
		
		final Matrix weightDeltas1 = errorInputDeriv1.removeLastElement().outerProduct(activation0);
		System.out.println("Weight gradients (layer 1): \n" + weightDeltas1);
		
		final Matrix newWeights1 = layer0.minus(weightDeltas1.transform(w -> w * eta0));
		System.out.println("New weights (layer 1): \n" + newWeights1);
	}
}
