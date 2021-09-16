#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <random>
#include "math.hpp"
#include "layer_dense.hpp"
#include "act_relu.hpp"
#include "softCCE.hpp"
#include "optimizer_sgd.hpp"
using namespace std;

int main(){

    Layer_Dense dense1(2, 64, 0, 0.0005, 0, 0.0005, 0.3);
    Layer_Dense dense2(64, 3, 0, 0, 0, 0, 0.3);
    Optimizer_SGD opti(1.0 , 0.001, 0.9);
    
    int numPoints = 1000;
    int dimension = 2;
    int numClasses = 3;

    vector<vector<double>> data (numClasses*numPoints);
    //each row is one data point, cols give coordinate vector
    vector<int> labels;
    createData(data, labels, numPoints, dimension, numClasses, 0);

    double currLoss = 0.0;
    for(int epochs = 1; epochs <= 1000; epochs++){

        Act_ReLU activation1(dense1.forward(data, false));
        SoftCCE loss(dense2.forward(activation1.forward(),false), labels);

        currLoss = meanLoss(loss.forward());

        if( epochs % 100 == 0){
            cout << "Epoch: " << epochs << " Loss: " << currLoss << "  Acc: " << loss.getAccuracy()  << " LR: " << opti.getLR();
            cout << "\n";
        }

        dense1.backward(activation1.backward(dense2.backward(loss.backward())));

        opti.preUpdateParams();
        opti.updateParams(dense1);
        opti.updateParams(dense2);
        opti.postUpdateParams();
        // cout << "reached";
    }

    for(int i = 0; i < data.size(); i ++){
        data[i].clear();
    }

    labels.clear();
    
    createData(data, labels, numPoints, dimension, numClasses, data.size() * 2);

    Act_ReLU activation1(dense1.forward(data, true));
    SoftCCE loss(dense2.forward(activation1.forward(), true), labels);
    currLoss = meanLoss(loss.forward());
    cout << "Tested Loss: " << currLoss << "  Acc: " << loss.getAccuracy();
}

