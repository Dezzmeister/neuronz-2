package dezzy.neuronz2.driver.meta;

import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuDeviceGetAttribute;
import static jcuda.driver.JCudaDriver.cuDeviceGetCount;
import static jcuda.driver.JCudaDriver.cuDeviceGetName;

import java.util.HashMap;
import java.util.Set;

import jcuda.driver.CUdevice;

public class DeviceManager {
	
	public static final Device[] getCUDADevices() {
		final int[] deviceCountPtr = {0};
		cuDeviceGetCount(deviceCountPtr);
		
		final int deviceCount = deviceCountPtr[0];
		final Device[] devices = new Device[deviceCount];		
		
		
		for (int i = 0; i < deviceCount; i++) {			
			final CUdevice device = new CUdevice();
			cuDeviceGet(device, i);
			
			final byte[] deviceName = new byte[1024];
			cuDeviceGetName(deviceName, deviceName.length, device);
			final String name = getNullTerminatedString(deviceName);
			
			final int[] ptr = {0};
			final Set<Integer> attributes = AttributeMap.map.keySet();
			final HashMap<Integer, Integer> attributeValues = new HashMap<Integer, Integer>();
			
			for (final int attribute : attributes) {
				cuDeviceGetAttribute(ptr, attribute, device);
				final int value = ptr[0];
				
				attributeValues.put(attribute, value);
			}
			
			devices[i] = new Device(name, attributeValues);
		}
		
		return devices;
	}	
	
	private static String getNullTerminatedString(final byte[] bytes) {
		final StringBuilder sb = new StringBuilder();
		
		for (int i = 0; i < bytes.length; i++) {
			final char c = (char) bytes[i];
			
			if (c == 0) {
				break;
			}
			
			sb.append(c);
		}
		
		return sb.toString();
	}
}
