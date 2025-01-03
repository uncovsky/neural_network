#pragma once
#include <cmath>
#include <istream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <fstream>
#include <iostream>
#include <math.h>

struct Loader{
    Loader() {}

    std::vector< std::vector< float > > load_vectors_from_csv(std::string path){
        std::vector<std::vector<float>> res;
        std::ifstream f(path, std::ios::in);

        if(!f.is_open()){
            std::cout << "Cannot open file " << path << "\n";
            return res;
        }

        std::string line;
        while (std::getline(f, line)) {

            std::vector<float> vector;
            float n;

            std::istringstream stream(line);

            while(stream >> n){
                vector.push_back(n);
                char ch;
                stream >> ch;
            }

            res.push_back(vector);
        }


        return res;
    }

    std::vector<int> load_labels_from_csv(std::string path) {
        std::vector<int> res;
        std::ifstream f(path, std::ios::in);

        if(!f.is_open()){
            std::cout<<"Cannot open file " << path << "\n";
            return res;
        }

        std::string line;
        while (std::getline(f, line)) {
            int n;
            std::istringstream stream(line);
            stream >> n;
            res.push_back(n);

        }


        return res;
    }


    void normalize(std::vector<std::vector<float>> &vectors, float mean, float sd) {
        for (std::vector<float>& vector : vectors) {
            for (float &n : vector) {
                n = (n-mean)/sd;
            }
        }
    }

    // return (mean, sd)
    std::tuple<float, float> normalize_dataset(std::vector<std::vector<float>> &vectors) {
        float mean = 0, variance = 0;

        float sum = 0.0;
        size_t count = 0;
        for (std::vector<float>& vector : vectors) {
            for (float n : vector) {
                sum += n;
                count += 1;
            }
        }

        mean = sum/count;

        float sq_sum = 0;
        for (std::vector<float>& vector : vectors) {
            for (float n : vector) {
                sq_sum += pow(n-mean, 2);
            }
        }

        variance = sq_sum/(count-1);
        float sd = std::sqrt(variance);

        normalize(vectors, mean, sd);

        return std::make_tuple(mean, sd);
    }


    void normalize_dataset(std::vector<std::vector<float>> &vectors, float mean, float sd) {
        normalize(vectors, mean, sd);
    }
};
