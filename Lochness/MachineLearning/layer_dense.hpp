#include <iostream>
#include <vector>
#include <string>
#include "math.hpp"
#pragma once

using namespace std;

class Layer_Dense{

    vector<vector<double>> dweights = {};
    vector<vector<double>> input = {};
    vector<vector<double>> dinput = {};
    vector<vector<double>> weights = {};
    vector<vector<double>> biases = {{}};
    vector<vector<double>> dbiases = {{}};
    vector<vector<double>> output = {};
    vector<vector<double>> weightMomentums = {};
    vector<vector<double>> biasMomentums = {};
    double weightsL1;
    double weightsL2;
    double biasL2; 
    double biasL1;
    double dropRate;
    string toString() const;
public:

Layer_Dense(int const &n_inputs, int const &n_neurons, double const &wL1=0, double const &wL2=0, double const &bL1=0, double const &bL2=0, double const &dr=0);
vector<vector<double>> forward(vector<vector<double>> inputs, bool predict);
vector<vector<double>> backward(vector<vector<double>> dvalues);
vector<vector<double>> getOutput() const;
vector<vector<double>> get_dbiases() const;
vector<vector<double>> get_dweights() const;
void updateWeight(int const &row, int const &col, double const &change);
void updateBias(int const &num, double const &change);
void updateMomentums(vector<vector<double>> const &weightUpdates, vector<vector<double>> const &biasUpdates);
vector<vector<double>> getWeightMomentums() const;
vector<vector<double>> getBiasMomentums() const;
friend ostream &operator<<(ostream &os, const Layer_Dense &d);


};