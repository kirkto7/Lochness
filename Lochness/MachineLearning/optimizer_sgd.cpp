#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "math.hpp"
#include "layer_dense.hpp"
#include "act_relu.hpp"
#include "softCCE.hpp"
#include "optimizer_sgd.hpp"
using namespace std;

Optimizer_SGD::Optimizer_SGD(double const &lr, double const &dcy, double const &mom ){
    default_learning_rate = lr;
    learning_rate = default_learning_rate;
    decay = dcy;
    momentum = mom;
}

void Optimizer_SGD::preUpdateParams(){
    if(decay != 0.0){
        learning_rate = default_learning_rate * ( 1.0 / (1.0 + (decay * interations)));
    }
}

void Optimizer_SGD::postUpdateParams(){
    interations += 1;
}

void Optimizer_SGD::updateParams(Layer_Dense &layer){
    weightUpdates = fillZeroes(layer.get_dweights());
    biasUpdates = fillZeroes(layer.get_dbiases());
    
    for(int i = 0; i < layer.get_dweights().size(); i++){
        for(int j = 0; j < layer.get_dweights()[i].size(); j++){
            if(momentum != 0){
                weightUpdates[i][j] = momentum * layer.getWeightMomentums()[i][j] - learning_rate * layer.get_dweights()[i][j]; 
                
            }else{
                weightUpdates[i][j] = -learning_rate * layer.get_dweights()[i][j];
            }
            layer.updateWeight(i, j, weightUpdates[i][j]);
        }
    }

    for(int i = 0; i < layer.get_dbiases().size(); i++){
        for(int j = 0; j < layer.get_dbiases()[i].size(); j++){
            if(momentum != 0){
                biasUpdates[i][j] = momentum * layer.getBiasMomentums()[i][j] - learning_rate * layer.get_dbiases()[i][j];
            }else{
                biasUpdates[i][j] = -learning_rate * layer.get_dbiases()[i][j];
            }
            layer.updateBias(j, biasUpdates[i][j]);
        }
    }

    layer.updateMomentums(weightUpdates, biasUpdates);

}

double Optimizer_SGD::getLR() const{
    return learning_rate;
}