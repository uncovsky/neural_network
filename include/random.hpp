#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>


class RNG {

    std::random_device rd;
    std::mt19937 gen{ std::random_device{}() };

    std::uniform_real_distribution< float > unif_dist;
    std::normal_distribution< float > norm_dist;

public:
    RNG() : rd(), gen() {
        gen.seed( rd() );
    }

    void seed() {
        gen.seed( rd() );
    }

    std::mt19937& get_gen(){
        return gen;
    }

    void seed( unsigned seed ) {
        gen.seed( seed );
    }

    std::vector< float > normal_vec( size_t count,
                                     float mu,
                                     float std ){

        std::vector< float > result( count );
        norm_dist = std::normal_distribution<float>(mu, std);

        std::generate( result.begin(), result.end(), [&](){ return norm_dist(gen); });

        return result;
    }

    std::vector< float > uniform_vec( size_t count,
                                      float min,
                                      float max ){

        std::vector< float > result( count );
        unif_dist = std::uniform_real_distribution<float>(min, max);

        std::generate( result.begin(), result.end(), [&](){ return unif_dist(gen); });

        return result;
    }
};

extern RNG rng;
