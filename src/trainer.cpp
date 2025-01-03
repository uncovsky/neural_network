#include "trainer.hpp"


Trainer::Trainer( NeuralNet *m, AdamOptimizer *opt, 
                  std::vector< std::vector< float > > d, 
                  std::vector< int > l ) : model(m), optimizer(opt), data(d), labels(l) {
    for ( size_t i = 0; i < data.size(); i++ ) {
        dataset.emplace_back( std::make_pair( data[i], labels[i]) );
    }
}


void Trainer::train( size_t epochs, size_t batch_size ) {

    // Reseed RNG to get deterministic shuffling during training
    gen.seed( 42 );

    auto train_start = std::chrono::high_resolution_clock::now();
    auto epoch_start = std::chrono::high_resolution_clock::now();
    auto epoch_end = std::chrono::high_resolution_clock::now();

    for ( size_t i = 0; i < epochs; i++ ) {

        epoch_start = std::chrono::high_resolution_clock::now();

        float total_l = 0;
        float total_samples = 0;
        float accuracy = 0;

        std::shuffle( dataset.begin(), dataset.end(), gen );

        for ( size_t sample = 0; sample < dataset.size() - batch_size; sample += batch_size ){

            total_samples += batch_size;

            std::vector< float > batch;
            std::vector< int > label_batch;

            // Make a batch of vectors and labels
            for ( size_t j = 0; j < batch_size; j++ ){
                batch.insert( batch.end(), dataset[sample+j].first.begin(), dataset[sample+j].first.end() );
                label_batch.push_back( dataset[sample+j].second );

            }

            // Pass matrix into model, get its predictions
            Matrix input( batch, dataset[sample].first.size(), batch_size );
            Matrix logits = model->forward( std::move(input) );
            auto preds = predictions( logits );

            // Calculate accuracy (on training set)
            for ( size_t pred_i = 0; pred_i < preds.size(); pred_i++ ) {
                if ( preds[pred_i] == label_batch[pred_i] ){ accuracy += 1; }
            }


            // Calculate loss_derivatives of softmax+CE, backprop
            Matrix loss_derivatives;
            total_l += cross_entropy_loss( logits, label_batch, loss_derivatives );
            model->backward( std::move(loss_derivatives) );

            // Take one step of GD
            optimizer->step();
        }

        epoch_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start).count();

        std::cout << "[Epoch: " << i + 1 << " / " << epochs << "; TIME: " << duration << " seconds.]\n";
        std::cout << "   Loss in epoch #" << i << " : " << total_l << "\n";
        std::cout << "   Accuracy in epoch #" << i << " : " << accuracy / (total_samples) << "\n";
    }

    auto duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - train_start).count();
    std::cout << "Training finished, total time: " << duration << ".\n";
}
