#pragma once
#include <cmath>

#include "model.hpp"


/*
 * Receives logits, computes derivative w.r.t each output for whole batch
 * returns the matrix of these derivatives in `derivatives` as well as the
 * value of the CE loss on this batch as a return value
 */

float cross_entropy_loss( const Matrix &logits,
                          const std::vector< int > &labels,
                          Matrix &derivatives ){

    Matrix out_derivatives = logits;

    float batch_size = logits.cols;
    float loss = 0;

    // Go over whole batch
    for ( size_t col = 0; col < batch_size; col++ ) {

        float denom = 0;

        // Assuming nonempty outputs
        float max = logits.at(0, col);


        // calculate max logit in this sample
        for ( size_t row = 0; row < logits.rows; row++ ) {
            max = std::max( max, logits.at( row , col ) );
        }

        // calculate denomiator of softmax sigma(y - maximum)
        // subtracting max from exponent makes computation a lot more stable
        for ( size_t row = 0; row < logits.rows; row++ ) {
            denom += std::exp( logits.at( row , col ) - max );
        }

        // calculate softmax, derivatives and loss for each output in this
        // sample
        for ( size_t row = 0; row < logits.rows; row++ ) {
            int match = ( labels[col] == row );
            float smax = std::exp( logits.at( row, col ) - max ) / denom;
            out_derivatives.at( row, col ) = smax - match;
            loss -= match * ( logits.at(row, col) - max  - std::log(denom));
        }
    }

    derivatives = std::move( out_derivatives );
    derivatives.multiply_scalar( 1/batch_size );

    return loss;
}


class AdamOptimizer {

    float _lr;

    // First & Second moment exp decay
    float _beta1;
    float _beta2;

    float _beta1t = 1.f;
    float _beta2t = 1.f;

    size_t timestep = 0;

    std::vector< Matrix* > _model_params;
    std::vector< Matrix* > _model_gradients;

    std::vector< std::unique_ptr< Matrix > > _first_moments;
    std::vector< std::unique_ptr< Matrix > > _second_moments;

public:

    AdamOptimizer( NeuralNet *m, float lr, float beta1, float beta2 );
    // Assumes Trainer called backward() with appropriate loss on the model,
    // collects gradients from `_model_gradients` and adjusts `_model_params`
    void step();
}



