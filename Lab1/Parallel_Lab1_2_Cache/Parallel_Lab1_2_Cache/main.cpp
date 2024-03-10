#include<iostream>
#include <chrono>
using namespace std;
void Cache(int* a, int sum, int n, int sum1, int sum2)
{
	for (int i = 0; i < n; i += 2) {
		sum1 += a[i];
		sum2 += a[i + 1];
	}
	sum = sum1 + sum2;
}
int main()
{
	int n = 10000;
	int* a = new int[n];
	int sum = 0;
	for (int i = 0; i < n; i++) {
		a[i] = i;
	}

	int sum1 = 0, sum2 = 0;
	auto beforeTime = std::chrono::steady_clock::now();
	for (int i = 0; i < n; i++) {
		Cache(a, sum, n, sum1, sum2);
	}
	auto afterTime = std::chrono::steady_clock::now();
	double time = std::chrono::duration<double>(afterTime - beforeTime).count();
	cout << "Cache Chain time=" << time << "seconds" << endl;
	return 0;
}