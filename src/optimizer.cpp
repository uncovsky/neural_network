#include "optimizer.hpp" 


/* Manually calculate softmax&CE loss given logits and correct labels.
 *
 * This is later used as input to the backpropagation algorithm in Trainer.
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


AdamOptimizer::AdamOptimizer( NeuralNet *m, float lr, 
                              float beta1, float beta2 ) : _lr( lr ),
                                                          _beta1( beta1 ), _beta2( beta2 ),
                                                          _model_params( m->params() ), _model_gradients( m->grads() ) {
        // Initialize first and second moment matrices
        for ( size_t i = 0; i < _model_params.size(); i++ ) {

            size_t rows = _model_params[i]->rows;
            size_t cols = _model_params[i]->cols;

            _first_moments.emplace_back( std::make_unique< Matrix >( rows, cols ) );
            _second_moments.emplace_back( std::make_unique< Matrix >( rows, cols ) );
        }

}


void AdamOptimizer::step() {

        _beta1t *= _beta1;
        _beta2t *= _beta2;

        float lr_t = _lr * std::sqrt( 1 - _beta2t ) / ( 1 - _beta1t );

        // Loop over all parameters of the network
        for ( size_t i = 0; i < _model_params.size(); i++ ){

            // Adjust first and second moment estimates
            Matrix gradient = *_model_gradients[i];
            Matrix gradient_square = *_model_gradients[i];

           gradient_square.cwise_product( gradient ).multiply_scalar( 1 - _beta2 );
            gradient.multiply_scalar( 1 - _beta1 );

            _first_moments[i]->multiply_scalar( _beta1 ).cwise_add( gradient );
            _second_moments[i]->multiply_scalar( _beta2 ).cwise_add( gradient_square );


            Matrix gradient_dir = *_second_moments[i];


            // calculate -alpha * m_t^ / ( sqrt(v_t^ + eps) )
            gradient_dir.apply( [&]( float x ){ return ( -1 * lr_t ) / ( std::sqrt( x ) + 1e-6 ); } );
            gradient_dir.cwise_product( *_first_moments[i] );

            _model_params[i]->cwise_add( gradient_dir );

        }
}
