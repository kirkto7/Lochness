#include <iostream>
#include <vector>
#include <string>
#pragma once
using namespace std;

class SoftCCE{

    int samples = 0;
    double acc = -1;
    vector<int> targets = {};
    vector<vector<double>> input = {};
    vector<vector<double>> dinput = {};
    vector<vector<double>> softOutput = {};
    vector<double> output = {};


public:

SoftCCE(vector<vector<double>> const &inp, vector<int> tar);
vector<double> forward();
vector<vector<double>> backward();
vector<vector<double>> getDerivative();
double getAccuracy();
};