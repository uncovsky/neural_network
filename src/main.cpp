#include <iostream>
#include <vector>

#include "activations.hpp"
#include "lingebra.hpp"
#include "model.hpp"
#include "optimizer.hpp"
#include "random.hpp"


int main() {


    int seed = 1;
    rng.seed( seed );

    auto layers = { std::make_shared< LinearLayer >( 784, 256, "relu", "he" ),
                    std::make_shared< LinearLayer >( 256, 10, "id", "he" ),
    };


    NeuralNet net( std::move( layers ) );

    float lr = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.999;

    size_t epochs = 30;
    size_t batch_size = 64;

    Loader load;
    std::vector< std::vector< float > > train_data = load.load_vectors_from_csv( "data/fashion_mnist_train_vectors.csv" );
    std::vector< int > train_labels = load.load_labels_from_csv( "data/fashion_mnist_train_labels.csv" );

    std::vector< std::vector< float > > test_data = load.load_vectors_from_csv( "data/fashion_mnist_test_vectors.csv" );
    std::vector< int > test_labels = load.load_labels_from_csv( "data/fashion_mnist_test_labels.csv" );

    auto [mean, sd] = load.normalize_dataset(train_data);
    load.normalize_dataset(test_data, mean, sd);


    AdamOptimizer opt( &net, lr, beta1, beta2 );

    Trainer trainer( &net, &opt, train_data, train_labels );
    trainer.train( epochs, batch_size );


    // Output predictions of the model
    std::ofstream test_output("test_predictions.csv");
    for ( size_t i = 0; i < test_data.size(); i++ ) {
        Matrix input(test_data[i], test_data[i].size(), 1);
        auto res = net.predict( std::move(input) );
        test_output << res[0] << "\n";
    }

    std::ofstream train_output("train_predictions.csv");
    for ( size_t i = 0; i < train_data.size(); i++ ) {
        Matrix input(train_data[i], train_data[i].size(), 1);
        auto res = net.predict( std::move(input) );
        train_output << res[0] << "\n";
    }

    return 0;
}
