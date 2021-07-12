package test.cuda;

import dezzy.neuronz2.driver.Init;
import dezzy.neuronz2.driver.meta.AttributeMap;
import dezzy.neuronz2.driver.meta.Device;
import dezzy.neuronz2.driver.meta.DeviceManager;
import jcuda.Pointer;
import jcuda.jcudnn.JCudnn;
import jcuda.runtime.JCuda;

@SuppressWarnings("unused")
public class RuntimeTest {
	
	public static void main(final String[] args) {
		final int[] versionPtr = new int[1];
		
		System.out.println(Init.CUDA_DRIVER_STATUS_CODE); // Force JCuda to initialize by loading the Init class
		JCuda.cudaDriverGetVersion(versionPtr);
		System.out.println(versionPtr[0]);
		// pointerTest();
		deviceTest();
	}
	
	private static final void deviceTest() {		
		final Device[] devices = DeviceManager.getCUDADevices();
		System.out.println("CUDA Devices: " + devices.length);
		System.out.println("================================");
		
		for (final Device device : devices) {
			System.out.println("Device Name: " + device.name);
			
			for (final int attribute : device.attributes.keySet()) {
				System.out.println(AttributeMap.map.get(attribute) + ": " + device.attributes.get(attribute));
			}
		}
		System.out.println("\n================================\n");
		System.out.println("cuDNN version: " + JCudnn.cudnnGetVersion());
	}
	
	private static final void pointerTest() {
		final Pointer pointer = new Pointer();
		JCuda.cudaMalloc(pointer, 4);
		System.out.println("Pointer: " + pointer);
		JCuda.cudaFree(pointer);
	}
}
