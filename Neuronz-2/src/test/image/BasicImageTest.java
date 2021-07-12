package test.image;

import java.io.IOException;

import dezzy.neuronz2.dataio.DoubleImage;

public class BasicImageTest {
	
	public static void main(final String[] args) throws IOException { 
		final DoubleImage imageIn = new DoubleImage("data/images/tee.png");
		final DoubleImage imageOut = new DoubleImage(imageIn.normPixels, imageIn.width, imageIn.height);
		imageOut.save("data/images/tee-out.png", "PNG");
	}
}
