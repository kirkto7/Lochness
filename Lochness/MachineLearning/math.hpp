#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <ctime>

using namespace std;

vector<vector<double>> dot_product(vector<vector<double>> v1, vector<vector<double>> v2); //dot product
vector<vector<double>> spiralData(); //dot product
string vector_string(vector<vector<double>> v1); //changes 2d vector into a string
void print_vector(vector<vector<double>> v1); //prints 2d vector
void print_arr(vector<int> v1); //prints 1d vector
void print_arr(vector<double> v1);
vector<vector<double>> add_vectors(vector<vector<double>> v1, vector<vector<double>> v2); //adds two vectors
vector<vector<double>> transpose(vector<vector<double>> v1); //switches collumns and rows of vector
double sum(vector<double> v1); //sums together a 1d vector
vector<vector<double>> sumVec(vector<vector<double>> v1, int axis); //sums together a 2d vector over chosen axis (0 = rows, 1 = columns)
double findMax(vector<double> v1); //finds the max num in a 1d vector
double findMaxInd(vector<double> v1); //finds the max num's index in a 1d vector
vector<double> expoVec(vector<double> v1); //exponentiaties a 1d vector
vector<double> subMax(vector<double> v1); //subtracts the maximum value from all nums in a 1d vector
double meanLoss(vector<double> v1); //takes the average loss across multiple batches
double accuracy(vector<vector<double>> v1, vector<int> targets); //calculates accuracy
vector<vector<double>> fillZeroes(vector<vector<double>> v1); //fills an identical array with zeroes
void createData(vector<vector<double>> &data, vector<int> &labels, int numPoints, int dimension, int numClasses, int adv);