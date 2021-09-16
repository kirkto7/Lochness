#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <ctime>
#include "math.hpp"
using namespace std;

vector<vector<double>> dot_product(vector<vector<double>> v1, vector<vector<double>> v2){
    if(v1[0].size() != v2.size() ){
        throw invalid_argument( "Mismatched Shape" );
    }
    vector<vector<double>> result = {};
    for(int k = 0; k < v1.size(); k++){ //fill result with vectors early
        vector<double> temp = {};
        result.push_back(temp);
    }
    for(int k = 0; k < v2[0].size(); k++){
        for(int i = 0; i < v1.size(); i++){
        double x = 0;
            for(int j = 0; j <v1[i].size(); j++){
                x += (v1[i][j] * v2[j][k]);
            }
        result[i].push_back(x);
        }
    }
    return result;
}

vector<vector<double>> add_vectors(vector<vector<double>> v1, vector<vector<double>> v2){
    if(v1[0].size() != v2[0].size()){
        throw invalid_argument( "Vectors need to be indentical in shape" );
    }
    vector<vector<double>> result = {};
    for(int k = 0; k < v1.size(); k++){ //fill result with vectors early
        vector<double> temp = {};
        result.push_back(temp);
    }
    for(int i = 0; i < v1.size(); i++){
        for(int j = 0; j < v1[i].size(); j++){
            double x = v1[i][j] + v2[0][j];
            result[i].push_back(x);
        }
    }
    return result;
}

vector<vector<double>> transpose(vector<vector<double>> v1){
    vector<vector<double>> result = {};
    for(int k = 0; k < v1[0].size(); k++){ //fill result with vectors early
        vector<double> temp = {};
        result.push_back(temp);
    }
    for(int i = 0; i < v1.size(); i ++){
        for(int j = 0; j < v1[0].size(); j++){
            result[j].push_back(v1[i][j]);
        }
    }
    return result;
}

string vector_string(vector<vector<double>> v1){
    string s = "";
    for(int i = 0; i < v1.size(); i++){
        for(int j = 0; j < v1[i].size(); j++){
            double x = v1[i][j];
            s += to_string(x);
            s += " | ";
        }
        s += "\n";
    }
    return s;
}

void print_vector(vector<vector<double>> v1){

    cout << vector_string (v1);
}

void print_arr(vector<int> v1){
    string s = "";
    for(int j = 0; j < v1.size(); j++){
            int x = v1[j];
            s += to_string(x);
            s += " | ";
        }
    cout << s;
    cout << "\n";
}

void print_arr(vector<double> v1){
    string s = "";
    for(int j = 0; j < v1.size(); j++){
            double x = v1[j];
            s += to_string(x);
            s += " | ";
        }
    cout << s;
    cout << "\n";
}

vector<vector<double>> activation_relu(vector<vector<double>> inputs){
    vector<vector<double>> output = {};
    for(int i = 0; i < inputs.size(); i++){
        vector<double> temp = {};
        output.push_back(temp);
        for(int j = 0; j < inputs[i].size(); j++){
            output[i].push_back(fmax(0,inputs[i][j]));
        }
    }
    return output;
}

double sum(vector<double> v1){
    double result = 0.0;
    for(int i = 0; i < v1.size(); i++){
        result += v1[i];
    }
    return result;
}

vector<vector<double>> sumVec(vector<vector<double>> v1, int axis){
    axis = fmin(1, fmax(0, axis));
    vector<vector<double>> results = {{}};
    if(axis == 1){
        v1 = transpose(v1);
    }
    for(int i = 0; i < v1.size(); i++){
        double sum = 0.0;
        for(int j = 0; j <v1[i].size(); j++){
            sum += v1[i][j];
        }
        results[0].push_back(sum);
    }
    return results;
}

double findMax(vector<double> v1){
    double highest = v1[0];
    for(int i = 0; i < v1.size(); i ++){
        if(v1[i] > highest){
            highest = v1[i];
        }
    }
    return highest;
}

double findMaxInd(vector<double> v1){
    double highest = 0;
    for(int i = 0; i < v1.size(); i ++){
        if(v1[i] > v1[highest]){
            highest = i;
        }
    }
    return highest;
}

vector<double> expoVec(vector<double> v1){
    vector<double> result = {};
    for(int i = 0; i < v1.size(); i++){
        result.push_back( exp (v1[i]) );
    }
    return result;
}

vector<double> subMax(vector<double> v1){
    vector<double> result = {};
    for(int i = 0; i < v1.size(); i++){
        double max = findMax(v1);
        result.push_back( v1[i] - max);
    }
    return result;
}

double meanLoss(vector<double> v1){
    double result = 0.0;
    for(int i = 0; i < v1.size(); i ++){
        result += v1[i];
    }
    result /= v1.size();
    return result;
}

double accuracy(vector<vector<double>> v1, vector<int> targets){
    vector<int> predictions = {};
    double result = 0.0;
    for(int i= 0; i < v1.size(); i ++){
        predictions.push_back(findMaxInd(v1[i]));
    }
    for(int i = 0; i < targets.size(); i++){
        if(predictions[i] == targets[i]){
            result += 1;
        }
    }
    result /=  targets.size();
    return result;
}

vector<vector<double>> fillZeroes(vector<vector<double>> v1){
    vector<vector<double>> result = v1;
    for(int i = 0; i < v1.size(); i++){
        for(int j = 0; j < v1[i].size(); j++){
            result[i][j] = 0;
        }
    }
    return result;
}

void createData(vector<vector<double>> &data, vector<int> &labels, int numPoints, int dimension, int numClasses, int adv){
    default_random_engine gen;
    gen.discard(adv);
    for(int i = 0; i < data.size(); i++){
        int classNum = i / numPoints; 
        // create data point in polar form

        // with evenly-spaced r values from 0 to 1
        double r = 1.0*i / numPoints - classNum;
        //and spiraling, slightly random theta values
        uniform_real_distribution<double> dis(0,1.0);
        double m =  dis(gen) * 1.4;\
         
        double theta = (classNum) * 2 + (6*r) + m;
        // convert polar to cartesion coordinates
        double x = r*cos(theta);
        double y = r*sin(theta);
        // assign into data array row
        data[i].push_back(x);
        data[i].push_back(y);

        // assign class labels, shifted to start at one
        labels.push_back(classNum);
    }
}
