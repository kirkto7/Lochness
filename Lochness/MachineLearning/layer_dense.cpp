#include <iostream>
#include <vector>
#include <string>
#include "math.hpp"
#include "layer_dense.hpp"
#include <random>
#include <ctime>

using namespace std;

Layer_Dense::Layer_Dense(int const &n_inputs, int const &n_neurons, double const &wL1, double const &wL2, double const &bL1, double const &bL2, double const &dr){
    default_random_engine gen (time(NULL));
    for(int i = 0; i < n_inputs; i++){
        vector<double> temp = {};
        weights.push_back(temp);
        for(int j = 0; j < n_neurons; j++){
            uniform_real_distribution<double> dis(-1.0,1.0);
            double m =  dis(gen);
            weights[i].push_back(m);
        }
    }
    for(int i = 0; i < n_neurons; i++){
        biases[0].push_back(0);
    }
    weightsL1 = wL1;
    weightsL2 = wL2;
    biasL1 = bL1;
    biasL2 = bL2;
    dropRate = dr;
    weightMomentums = fillZeroes(weights);
    biasMomentums = fillZeroes(biases);
}

vector<vector<double>> Layer_Dense::forward(vector<vector<double>> inp, bool predict){
    output = add_vectors(dot_product(inp, weights),biases);
    input = inp;
    // if(!predict){
    // default_random_engine gen;
    // gen.discard((int)(weights[0][0] * 1000));
    // bernoulli_distribution dis (dropRate);
    //     for(int i = 0; i < output.size(); i ++){
    //         for(int j = 0; j < output[i].size(); j ++){
    //             if(dis(gen)){
    //                 output[i][j] *= 1.0 / (1.0 - dropRate);
    //             }else{
    //                 output[i][j] = 0;
    //             }
    //         }
    //     }
    // }
    return output;
}

vector<vector<double>> Layer_Dense::backward(vector<vector<double>> dvalues){
    dweights = dot_product(transpose(input), dvalues);
    dbiases = sumVec(dvalues, 1);
    if(weightsL1 > 0 || weightsL2 > 0){
        for(int i = 0; i < dweights.size(); i++){
            for(int j = 0; j < dweights[i].size(); j++){
                double thisWeight = weights[i][j];
                if(weightsL1 > 0){
                    if(thisWeight != 0){
                        dweights[i][j] += weightsL1 * (thisWeight / abs(thisWeight));

                    }else{
                        dweights[i][j] += weightsL1;
                    }
                }
                if(weightsL2 > 0){
                    dweights[i][j] += 2 * weightsL2 * thisWeight;
                }

            }
        }
    }

    if(biasL1 > 0 || biasL2 > 0){
        for(int i = 0; i < dbiases[0].size(); i++){
            double thisBias = biases[0][i];
            if(biasL1 > 0){
                if(thisBias != 0){
                    dbiases[0][i] += biasL1 * (thisBias / abs(thisBias));

                }else{
                    dbiases[0][i] += biasL1;
                }
            }
            if(biasL2 > 0){
                dbiases[0][i] += 2 * biasL2 * thisBias;
            }

        }
    }

    dinput = dot_product(dvalues, transpose(weights));

    return dinput;
}

vector<vector<double>> Layer_Dense::getOutput() const{
    return output;
}

vector<vector<double>> Layer_Dense::get_dbiases() const{
    return dbiases;
}

vector<vector<double>> Layer_Dense::get_dweights() const{
    return dweights;
}

void Layer_Dense::updateWeight(int const &row, int const &col, double const &change){
    weights[row][col] += change;
}

void Layer_Dense::updateBias(int const &num, double const &change){
    biases[0][num] += change;
}

void Layer_Dense::updateMomentums(vector<vector<double>> const &weightUpdates, vector<vector<double>> const &biasUpdates){
    weightMomentums = weightUpdates;
    biasMomentums = biasUpdates;
}

vector<vector<double>> Layer_Dense::getWeightMomentums() const{
    return weightMomentums;
}

vector<vector<double>> Layer_Dense::getBiasMomentums() const{
    return biasMomentums;
}

string Layer_Dense::toString() const{
    string s = "Weights: \n";
    s += vector_string(weights);
    s+= "\n";
    s += "Biases: \n";
    s+= vector_string(biases);
    s+= "\n";
    return s;
}

ostream &operator<<(ostream &os, const Layer_Dense &l){
    os << l.toString();
    return os;

}

