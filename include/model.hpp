#pragma once

#include "activations.hpp"
#include "lingebra.hpp"
#include "random.hpp"

#include <limits>
#include <map>
#include <memory>


// Mode of weight matrix initialization (see model.cpp for implementation)
enum class InitializationMode { Uniform, He, Pytorch };


// Get predictions for a model
std::vector< size_t > predictions ( const Matrix& logits );


/*
 * Class that implements a fully connected layer.
 * Stores matrices, implements forward and backward pass.
 *
 * Stores backpropagation related info - such as derivatives w.r.t. outputs of
 * layer, weights, potentials.
 *
 */

class LinearLayer {

public:

    // Matrices, activation function
    Matrix _weights;
    Matrix _bias;
    ActivationFunction *_act;

    // Signals whether _bias is allocated & initialized
    bool has_bias;

    // do not store backpropagation info (inference mode)
    bool evaluation = false;

    // Write debug information during FP/BP
    bool debug_output = false;

    /*
     *  Forward pass related information
     *
     *  Inputs to forward() and the resulting outputs of next layer
     */
    Matrix _inputs;
    Matrix _outputs;

    /*
     * Backprop related information (stored during FP if `evaluation` is false)
     */

    // derivatives of potentials of the following layer, i.e. \sigma'(potentials2)
    Matrix _potentials_derivatives;

    // Resulting derivatives of the previous layer (after calling backward())
    Matrix _prev_derivatives;


    /*
     * Data for optimizer
     */

    // gradients w.r.t weights
    Matrix _weight_gradients;

    // gradients w.r.t biases 
    Matrix _bias_gradients;

    LinearLayer( size_t input_dim, size_t output_dim, 
                 std::string activation, 
                 std::string init_mode = "he",
                 bool bias=true );

    /*
     * Helper functions
     */
    void initialize_weights( size_t m, size_t n, InitializationMode mode);
    void initialize_info( size_t m, size_t n );
    void set_evaluation( bool mode );

    /*
     * Optimizer related getters
     */
    std::vector< Matrix* > get_params();
    std::vector< Matrix* > get_grads();


    /*
     * Core functionality
     */
    Matrix forward( Matrix&& inputs );
    Matrix backward( Matrix&& derivatives );
};


/*
 * Glorified container for Layers
 */
class NeuralNet {

    std::vector< std::shared_ptr< LinearLayer > > _layers;

public:
    NeuralNet();
    NeuralNet( std::vector< std::shared_ptr<LinearLayer > > &&layers);

    void evaluation();
    void training();

    std::vector< Matrix* > params();
    std::vector< Matrix* > grads();

    Matrix forward( Matrix &&input );
    void backward( Matrix&& derivatives );
    std::vector< size_t > predict( Matrix&& input );
};

