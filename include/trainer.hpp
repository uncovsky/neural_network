#pragma once
#include "model.hpp"
#include <chrono>


class Trainer {

    // Generator that governs the data shuffling during training
    std::mt19937 gen{ std::random_device{}() };

    // Neural net that is optimized
    NeuralNet *model;

    // Initialized optimizer object
    AdamOptimizer *optimizer;

    std::vector< std::vector< float > > data;
    std::vector< int > labels;


    // Dataset of vectors : labels, data + labels zipped together
    std::vector< std::pair< std::vector< float >, int > > dataset;

public:

    Trainer( NeuralNet *m, AdamOptimizer *opt, 
            std::vector< std::vector< float > > d, std::vector< int > l );

    void train( size_t epochs, size_t batch_size );
};
