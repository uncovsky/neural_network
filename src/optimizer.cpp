#include "optimizer.hpp" 



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
