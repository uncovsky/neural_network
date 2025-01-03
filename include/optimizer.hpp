#pragma once
#include <cmath>

#include "model.hpp"


float cross_entropy_loss( const Matrix &logits,
                          const std::vector< int > &labels,
                          Matrix &derivatives );
/*
 * Receives logits, computes derivative w.r.t each output for whole batch
 * returns the matrix of these derivatives in `derivatives` as well as the
 * value of the CE loss on this batch as a return value
 */

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
};



