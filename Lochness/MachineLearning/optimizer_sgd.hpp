#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "math.hpp"
#include "layer_dense.hpp"
#include "act_relu.hpp"
#include "softCCE.hpp"
using namespace std;

class Optimizer_SGD{

double interations = 0;
double default_learning_rate = 0;
double learning_rate = 0;
double decay = 0;
double momentum = 0;
vector<vector<double>> weightUpdates = {};
vector<vector<double>> biasUpdates = {};

public:

Optimizer_SGD(double const &lr,double const &dcy, double const &mom);
void updateParams(Layer_Dense &layer);
void preUpdateParams();
void postUpdateParams();
double getLR() const;
};