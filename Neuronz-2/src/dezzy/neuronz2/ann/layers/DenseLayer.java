package dezzy.neuronz2.ann.layers;

import dezzy.neuronz2.arch.layers.Layer;
import dezzy.neuronz2.math.constructs.Matrix;
import dezzy.neuronz2.math.constructs.Vector;

public class DenseLayer implements Layer<Vector, Vector> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -289894409148356984L;
	
	private Matrix weights;
	
	public DenseLayer(final Matrix _weights) {
		weights = _weights;
	}

	@Override
	public Vector forwardPass(final Vector prevActivations) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Vector backprop(final Vector errorOutputDeriv, final boolean isFirstLayer) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void update(final double learningRate) {
		// TODO Auto-generated method stub
		
	}
	
}
