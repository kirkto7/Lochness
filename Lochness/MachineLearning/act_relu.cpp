#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "math.hpp"
#include "layer_dense.hpp"
#include "act_relu.hpp"
using namespace std;

Act_ReLU::Act_ReLU(vector<vector<double>> const &inp){
    input = inp;
}

vector<vector<double>> Act_ReLU::forward(){
    output.clear();
    for(int i = 0; i < input.size(); i++){
        vector<double> temp = {};
        output.push_back(temp);
        for(int j = 0; j < input[i].size(); j++){
            output[i].push_back(fmax(0,input[i][j]));
        }
    }
    return output;
}
vector<vector<double>> Act_ReLU::backward(vector<vector<double>> const &dvalues){
    dinput = dvalues;
    for(int i = 0; i < dinput.size(); i++){
        for(int j = 0; j <dinput[i].size(); j++){
            if(input[i][j] <= 0){
                dinput[i][j] = 0;
            }
        }
    }
    return dinput;
}
vector<vector<double>> Act_ReLU::getDerivative(){
    return dinput;
}
