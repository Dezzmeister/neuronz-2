package dezzy.neuronz2.math.utility;

/**
 * Thrown when the dimension(s) of a value do not match the expected dimension(s).
 *
 * @author Joe Desmond
 */
public class DimensionMismatchException extends IllegalArgumentException {

	/**
	 * 
	 */
	private static final long serialVersionUID = -950172572524496584L;
	
	/**
	 * Constructs a DimensionMismatchException with the given message.
	 * 
	 * @param message exception message
	 */
	public DimensionMismatchException(final String message) {
		super(message);
	}
}
