package dezzy.neuronz2.driver.meta;

import java.util.Map;

public final class Device {
	public final String name;
	public final Map<Integer, Integer> attributes;
	
	public Device(final String _name, final Map<Integer, Integer> _attributes) {
		name = _name;
		attributes = _attributes;
	}
}
