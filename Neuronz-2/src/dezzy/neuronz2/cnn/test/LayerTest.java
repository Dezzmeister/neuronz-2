package dezzy.neuronz2.cnn.test;

import dezzy.neuronz2.cnn.layers.ConvolutionLayer2;
import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.constructs.Tensor3;
import dezzy.neuronz2.math.constructs.Tensor4;
import dezzy.neuronz2.math.constructs.Vector;

public final class LayerTest {
	
	public static final void main(final String[] args) {
		test1();
	}
	
	private static final void test1() {
		final Matrix kernel = new Matrix(new double[][] {
			{-1., 0.5},
			{0.2, 3.}
		});
		
		final Matrix image = new Matrix(new double[][] {
			{4, 3, 2, 1},
			{1, 2, 0, 4},
			{6, 1, 3, 2},
			{7, 8, 9, 9}
		});
		
		final ConvolutionLayer2 layer0 = new ConvolutionLayer2(new Tensor4(new Tensor3(kernel)), new Vector(0));
		final Tensor3 forwardResult = layer0.forwardPass(new Tensor3(image));
		System.out.println(forwardResult.getLayer(0));
		final Tensor3 transformed = forwardResult.transform(d -> d / 4);
		
		final Tensor3 backwardResult = layer0.backprop(transformed, false);
		System.out.println("\n" + backwardResult);
	}
}
