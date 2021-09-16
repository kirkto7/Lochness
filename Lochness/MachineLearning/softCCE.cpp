#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "math.hpp"
#include "softCCE.hpp"
using namespace std;

SoftCCE::SoftCCE(vector<vector<double>> const &inp, vector<int> tar){
    input = inp;
    samples = inp.size();
    targets = tar;
}

vector<double> SoftCCE::forward(){
    //softmax activation
    output.clear();
    softOutput.clear();
    for(int i = 0; i < input.size(); i ++){
        vector<double> temp = subMax(input[i]);
        temp = expoVec(temp);
        double total = sum(temp);
        for(int j = 0; j < temp.size(); j ++){
            temp[j] = temp[j] / total;
        }
        softOutput.push_back(temp);
    }

    acc = accuracy(softOutput, targets);

    //calculate loss, softOutput as input (list of confidences)
    for(int i = 0; i < softOutput.size(); i ++){
        double tnum = fmax(1/10000000 , (fmin( softOutput[i][targets[i]], (1 - 1/10000000))));
        double num = -1 * log(tnum);
        output.push_back(num);
    }
    return output;
}

vector<vector<double>> SoftCCE::backward(){
    dinput = softOutput;
    for(int i = 0; i < targets.size(); i++){
        dinput[i][targets[i]] -= 1;
    }
    for(int i = 0; i < dinput.size(); i ++){
        for(int j = 0; j <dinput[i].size(); j++){
            dinput[i][j] /= targets.size();
        }
    }
    return dinput;
}

vector<vector<double>> SoftCCE::getDerivative(){
    return dinput;
}

double SoftCCE::getAccuracy(){
    return acc;
}

