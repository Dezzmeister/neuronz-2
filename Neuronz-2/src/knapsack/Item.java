package knapsack;


/**
 * A box in the knapsack problem, has a weight and a value
 *
 * @author Joe Desmond
 */
public class Item {
	
	/**
	 * Value (must be maximized by neural network)
	 */
	final float value;
	
	/**
	 * Weight
	 */
	final float weight;
	
	/**
	 * Constructs a Box with the given weight and value.
	 * 
	 * @param _value value
	 * @param _weight weight
	 */
	public Item(final float _value, final float _weight) {
		value = _value;
		weight = _weight;
	}
	
	/**
	 * Adds the weight and value of this item to another and returns the result.
	 * Does not mutate this or the other item.
	 * 
	 * @param other other item
	 * @return a new item
	 */
	public final Item plus(final Item other) {
		return new Item(value + other.value, weight + other.weight);
	}
}
