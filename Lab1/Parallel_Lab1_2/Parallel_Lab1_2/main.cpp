#include<iostream>
#include <chrono>
using namespace std;
void Common(int* a, int sum,int n)
{
	for (int i = 0; i < n; i++)
		 sum += a[i];
}
int main()
{
	int n = 10000;
	int* a = new int[n];
	int sum = 0;
	for (int i = 0; i < n; i++) {
		a[i] = i;
	}

	auto beforeTime = std::chrono::steady_clock::now();
	for (int i = 0; i < n; i++) {
		Common(a, sum, n);
	}
	auto afterTime = std::chrono::steady_clock::now();
	double time = std::chrono::duration<double>(afterTime - beforeTime).count();
	cout << "Common time=" << time << "seconds" << endl;
	return 0;
}