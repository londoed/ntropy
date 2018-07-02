class Ntropy
  getter :shape, :outputs
  property :weights, :weight_update_values

  DEFAULT_TRAINING_OPTIONS = {
    :max_iterations => 1000
    :error_threshold => 0.01
  }

  def initialize(shape)
    @shape = shape
  end

  def run(inp)
    # Input to this method represents the output of the first layer
    @outputs = [inp]
    set_initial_weight_values if @weights.nil?

    # Calculate output of neurons in subsequent layers
    1.upto(output_layer).each do |layer|
      source_layer = layer - 1
      source_outputs = @outputs[source_layer]
      @outputs[layer] = @weights[layer].map do |n_weights|
        # Inputs to this neuron are the outputs from the neurons in the source layer multiplited by weights
        inp = n_weights.map.with_index do |weight, i|
          source_output = source_outputs[i] || 1 # If no output, this is the bias neuron
          weight * source_output
        end
        input_sum = inp.reduce(:+)
        # The activated output of this neuron (using sigmoid)
        sigmoid(sum_of_inputs)
      end
    end
    # Outputs of neurons in the last layer is final result
    return @outputs[output_layer]
  end

  def train(inputs, expected_outputs, opts = {} of GenNum => GenNum)
    opts = DEFAULT_TRAINING_OPTIONS.merge(opts)
    error_threshold = opts[:error_threshold]
    log_every = opts[:log_every]
    iter, error = 0, 0

    set_initial_weight_update_values if @weight_update_values.nil?
    set_weight_changes_to_zeros
    set_previous_gradients_to_zeros

    while iter < opts[:max_iterations]
      iter += 1
      error = train_on_batch(inputs, expected_outputs)

      if log_every && (iter % log_every == 0)
        puts "#{iter} #{(error * 100).round(2)}% mse"
      end
      break if error_threshold && (error < error_threshold)
    end
    {error => error.round(5), iterations => iter, below_error_threshold => (error < error_threshold)}
  end

  private def train_on_batch(inputs, expected_outputs)
    total_mse = 0
    set_gradients_to_zeros

    inputs.each.with_index do |inp, i|
      run(inp)
      training_error = calculate_training_error(expected_outputs[i])
      update_gradients(training_error)
      total_mse += mean_squared_error(training_error)
    end
    update_weights
    return total_mse / inp.length.to_f # Average mse for batch
  end

  private def calculate_training_error(ideal_output)
    @outputs[output_layer].map.with_index do |output, i|
      return output - ideal_output[i]
    end
  end

  private def update_gradients(training_error)
    deltas = {} of GenNum => GenNum
    # Starting from output layer and working backwards, backpropagating the training error
    output_layer.downto(1).each do |layer|
      deltas[layer] = [] of GenNum

      @shape[layer].times do |n|
        neuron_error = if layer == output_layer
          -training_error[n]
        else
          target_layer = layer + 1
          weighted_target_deltas = deltas[target_layer].map.with_index do |target_delta, target_neuron|
            target_weight = @weights[target_layer][target_neuron][n]
            target_delta * target_weight
          end
          weighted_target_deltas.reduce(:+)
        end
        output = @outputs[layer][n]
        activation_derivative = output * (1.0 - output)
        delta = deltas[layer][n] = neuron_error * activation_derivative
        # Gradient for each of this neuron's incoming weights is calculated
        # The last output from incoming source neuron from -1 layer
        # Times this neuron's delta
        source_neurons = @shape[layer - 1] + 1 # Account for bias neuron
        source_outputs = @outputs[layer - 1]
        gradients = @gradients[layer][n]

        source_neurons.times do |source_neuron|
          source_output = source_outputs[source_neuron] || 1 # If no output, this is the bias neuron
          gradient = source_output * delta
          gradients[source_neuron] += gradient # Accumulate gradients from batch
        end
      end
    end

    MIN_STEP = Math.exp(-6)
    MAX_STEP = 50

    # The calculated gradients for the batch can be used to update the weights
    # RPROP algorithm
    def update_weights
      1.upto(output_layer) do |layer|
        source_layer = layer - 1
        source_neurons = @shape[source_layer] + 1 # Account for bias neuron

        @shape[layer].times do |n|
          source_neurons.times do |source_neuron|
            weight_change = @weight_changes[layer][n][source_neuron]
            weight_update_value = @weight_update_values[layer][n][source_neuron]
            gradient = -@gradient[layer][n][source_neuron]
            previous_gradient = @previous_gradients[layer][n][source_neuron]
            c = sign(gradient * previous_gradient)

            case c
            when 1 then # No sign change; accelerate gradient descent
              weight_update_value = [weight_update_value * 1.2, MAX_STEP].min
              weight_change = -sign(gradient) * weight_update_value
            when -1 then # Sign change; jumped over local min
              weight_update_value = [weight_update_value * 0.5, MIN_STEP].max
              weight_change = -weight_change # Roll back previous weight change
              gradient = 0 # Won't trigger sign change on update
            when 0 then
              weight_change = -sign(gradient) * weight_update_value
            end
            @weights[layer][n][source_neuron] += weight_change
            @weight_changes[layer][n][source_neuron] = weight_change
            @weight_update_values[layer][n][source_neuron] = weight_update_value
            @previous_gradients[layer][n][source_neuron] = gradient
          end
        end
      end
    end

    def set_weight_changes_to_zeros
      @weight_changes = build_connection_matrices { 0.0 }
    end

    def set_gradients_to_zeros
      @gradients = build_connection_matrices { 0.0 }
    end

    def set_previous_gradients_to_zeros
      @previous_gradients = build_connection_matrices { 0.0 }
    end

    def set_initial_weight_update_values
      @weight_update_values = build_connection_matrices { 0.1 }
    end

    def set_initial_weight_values
      # Initialize all weights to random float value
      @weights = build_connection_matrices { rand(-0.5..0.5) }
      # Update weights for first hidden layer
      beta = 0.7 * @shape[1]**(1.0 / @shape[0])
      @shape[1].times do |n|
        weights = @weights[1][n]
        norm = Math.sqrt(weights.map { |w| w**2 }.reduce(:+))
        updated_weights = weights.map { |w| (beta * w) / norm }
        @weights[1][n] = updated_weights
      end
    end

    def build_connection_matrices
      1.upto(output_layer).inject do |hsh, layer|
        # Number of incoming connections to each neuron in this layer
        source_neurons = @shape[layer - 1] + 1 # == number of neurons in previous layer + a bias neuron
        matrix = Array.new(@shape[layer]) do |n|
          Array.new(source_neurons) { yield }
        end
        hsh[layer] = matrix
        return hsh
      end
    end

    def output_layer
      @shape.length - 1
    end

    def sigmoid(x)
      1 / (1 + Math.exp(-x))
    end

    def mean_squared_error(errors)
      errors.map { |e| e**2 }.reduce(:+) / errors.length.to_f
    end

    ZERO_TOLERANCE = Math.exp(-16)

    def sign(x)
      if x > ZERO_TOLERANCE
        x *= 1
      elsif x < -ZERO_TOLERANCE
        x *= -1
      else
        x *= 0
      end
    end

    def marshal_dump
      [@shape, @weights, @weight_update_values]
    end

    def marshal_load(array)
      @shape, @weights, @weight_update_values = array
    end
  end
end
