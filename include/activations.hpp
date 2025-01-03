#pragma once

#include <algorithm>
#include <string>

class ActivationFunction {

public:
    virtual ~ActivationFunction() {};

    // apply activation on x 
    virtual float forward( float x ) = 0;
    
    // apply derivative of activation on x
    virtual float backward( float x ) = 0;

    virtual std::string desc() = 0;
};


class Identity : public ActivationFunction {

public:

    static Identity* get_instance(){
        static Identity id;
        return &id;
    }
    
    float forward( float x ) override {
        return x;
    }

    float backward( float ) override {
        return 1;
    }


    std::string desc() override {
        return "Id";
    }
};

class RELU : public ActivationFunction {

public:

    static RELU* get_instance(){
        static RELU relu;
        return &relu;
    }
    
    float forward( float x ) override {
        return std::max(0.f, x);
    }

    // TODO: remove branching ideally
    float backward( float x ) override {
        return x > 0 ? 1.f: 0.f;
    }

    std::string desc() override {
        return "ReLU";
    }
};


