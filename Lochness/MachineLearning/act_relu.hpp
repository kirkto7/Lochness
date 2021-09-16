#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "math.hpp"
#include "layer_dense.hpp"
#pragma once
using namespace std;

class Act_ReLU{


    vector<vector<double>> input = {};
    vector<vector<double>> dinput = {};
    vector<vector<double>> output = {};

public:

Act_ReLU(vector<vector<double>> const &inp);
vector<vector<double>> forward();
vector<vector<double>> backward(vector<vector<double>> const &dvalues);
vector<vector<double>> getDerivative();


};