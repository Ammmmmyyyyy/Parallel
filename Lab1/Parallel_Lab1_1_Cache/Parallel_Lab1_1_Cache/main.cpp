#include<iostream>
#include <chrono>
using namespace std;
void Cache(double** b, double* a, double* sum, int n)
{
	for (int i = 0; i < n; i++)
		sum[i] = 0.0;
	for (int j = 0; j < n; j++)
		for (int i = 0; i < n; i++)
			sum[i] += b[j][i] * a[j];
}
int main()
{
	int n = 1000;
	double* sum = new double[n], * a = new double[n];
	double** b = new double* [n];
	for (int i = 0; i < n; ++i) {
		b[i] = new double[n];
	}
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			b[i][j] = i + j;
		}
	}
	for (int i = 0; i < n; i++) {
		a[i] = i;
	}

	auto beforeTime = std::chrono::steady_clock::now();
	for (int i = 0; i < n; i++) {
		Cache(b, a, sum, n);
	}
	auto afterTime = std::chrono::steady_clock::now();
	double time = std::chrono::duration<double>(afterTime - beforeTime).count();
	cout << "Cache time=" << time << "seconds" << endl;
	return 0;
}