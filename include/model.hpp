#pragma once

#include "activations.hpp"
#include "lingebra.hpp"
#include "random.hpp"

#include <limits>
#include <map>
#include <memory>

// Do uniform (-1/2, 1/2), He init, and pytorch init (see linear layer docs) 
enum class InitializationMode { Uniform, He, Pytorch };

std::map< std::string, ActivationFunction* > activation_map = { 
        { "relu" , RELU::get_instance() },
        { "id" , Identity::get_instance() }
};

std::map< std::string, InitializationMode > init_map = { 
        { "he", InitializationMode::He },
        { "pytorch", InitializationMode::Pytorch },
        { "uniform", InitializationMode::Uniform }
};

/*
 *
 *  Could potentially move this to a Layer superclass, then implement
 *  Conv layers / Dropout etc.
 *
 */
class LinearLayer {

public:
    Matrix _weights;
    Matrix _bias;

    ActivationFunction *_act;

    bool has_bias;

    // do not store backpropagation info
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

    void initialize_weights( size_t m, size_t n, InitializationMode mode) {

        std::vector< float > weights;

        if ( mode == InitializationMode::Uniform ) {
            weights = rng.uniform_vec( m * n, -0.5, 0.5 );
        }

        else if ( mode == InitializationMode::He ) {
            weights = rng.normal_vec( m * n, 0.0, 1.0 / std::sqrt(n) );
        }

        // pytorch init
        else {
            weights = rng.uniform_vec( m * n, -1.0 / std::sqrt(n), 1.0 / std::sqrt(n) );
        }

        _weights = Matrix(std::move( weights ), m, n);

        if ( has_bias ) { 
            // init to small value >0
            _bias = Matrix(m, 1); 
            _bias.add_scalar(0.01);
        }
    }


    LinearLayer( size_t input_dim, size_t output_dim, 
                 std::string activation, 
                 std::string init_mode = "he",
                 bool bias=true ) {
        has_bias = bias;
        _act = activation_map[ activation ];
        initialize_weights( output_dim, input_dim, init_map[init_mode] );
    }


    std::vector< Matrix* > get_params() {
        std::vector< Matrix* > params = { &_weights };
        if ( has_bias ) {
            params.push_back( &_bias );
        }

        return params;
    }

    std::vector< Matrix* > get_grads() {
        std::vector< Matrix* > grads = { &_weight_gradients };
        if ( has_bias ) {
            grads.push_back( &_bias_gradients );
        }

        return grads;
    }

    void initialize_info( size_t m, size_t n ) {
        _weight_gradients = Matrix(m, n);
        _bias_gradients = Matrix(m, n);
    }

    void set_evaluation( bool mode ) {
        evaluation = mode;
    }


    Matrix forward( Matrix&& inputs ){
        
        auto forward_act = [&](float x){ return _act->forward(x); };
        auto backward_act = [&](float x){ return _act->backward(x); };


        if ( debug_output ) {
            std::cout << "INPUT to forward call:\n";
            inputs.print();
        }

        _outputs = _weights.mult( inputs ).cwise_add( _bias );

        if ( !evaluation ){
            _inputs = std::move(inputs);
            _potentials_derivatives = _outputs;
            _potentials_derivatives.apply( backward_act );

        }
        
        // Calculate outputs of the layer
        _outputs.apply( forward_act );

        if ( debug_output ) {
            std::cout << "result of forward call:\n";
            _outputs.print();
        }
        
        return _outputs;
    }

    /*
     * Given a derivative w.r.t to output of the following layer,
     * compute derivatives w.r.t outputs y of the previous one.
     *
     * Also use the input derivatives to calculate and save gradients for
     * `_weights` and `_bias`.
     */
    Matrix backward( Matrix&& derivatives ){

        if ( debug_output ) {
            std::cout << "INPUT to backward call:\n";
            derivatives.print();
        }
        
        derivatives.cwise_product( _potentials_derivatives );

        // Derivatives w.r.t outputs for the previous layer
        _prev_derivatives = _weights.transpose().mult( derivatives );

        if ( debug_output ) {
            std::cout << "previous derivatives backward call:\n";
            _prev_derivatives.print();
        }

        // Now calculate derivatives w.r.t weights & biases
        // The formula is prev = next derivatives * potentials of
        // outputs * tranposed input matrix
        _weight_gradients = derivatives.mult( _inputs.transpose() );

        // Biases are just sums of the losses, no multiplication by inputs required
        _bias_gradients = std::move( derivatives.row_reduce() );

        if ( debug_output ) {
            std::cout << "weight and bias gradients backward call:\n";
            _weight_gradients.print();
            _bias_gradients.print();
        }

        return _prev_derivatives;
    }

};


// Predict labels from logits
std::vector< size_t > predictions ( const Matrix& logits ) {

    std::vector< size_t > predictions;

    for ( size_t col = 0; col < logits.cols; col++ ){
        size_t argmax = 0;
        float max = -std::numeric_limits<double>::infinity();

        for ( size_t row = 0; row < logits.rows; row++ ){
            if ( logits.at( row, col ) > max ){
                max = logits.at( row, col );
                argmax = row;
            }
        }

        predictions.push_back( argmax );
    }

    return predictions;
}


/*
 * Glorified container for Layers
 */
class NeuralNet {

    std::vector< std::shared_ptr< LinearLayer > > _layers;

public:
    NeuralNet() : _layers() { }
    NeuralNet( std::vector< std::shared_ptr<LinearLayer > > &&layers) : _layers(std::move(layers)) {};

    Matrix forward( Matrix &&input ){

        for ( size_t i = 0; i < _layers.size(); i++ ) {
           input = _layers[i]->forward( std::move(input) );
        }

        return input;
    }

    // calculate gradients for all weights
    void backward( Matrix&& derivatives ){
        for ( size_t i = _layers.size(); i > 0; --i ) {
           derivatives = _layers[i-1]->backward( std::move(derivatives) );
        }
    }

    void evaluation() {
        for ( auto& layer : _layers ) {
            layer->set_evaluation( true );
        }
    }

    std::vector< Matrix* > params() {
        std::vector< Matrix* > res;
        for ( size_t i = 0; i < _layers.size(); i++ ){
            for ( auto ptr : _layers[i]->get_params() ){
                res.push_back(ptr);
            }
        }

        return res;
    }

    std::vector< Matrix* > grads() {
        std::vector< Matrix* > res;
        for ( size_t i = 0; i < _layers.size(); i++ ){
            for ( auto ptr : _layers[i]->get_grads() ){
                res.push_back(ptr);
            }
        }

        return res;
    }
    void training(){
        for ( auto& layer : _layers ) {
            layer->set_evaluation( false );
        }
    }

    std::vector< size_t > predict( Matrix&& input ) {
        evaluation();
        auto result = forward( std::move(input) );
        return predictions( result );
    }

};

