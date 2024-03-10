#include<iostream>
#include <chrono>
using namespace std;
void Recursion(int* a, int n)
{
	if (n == 1)
		return;
	else
	{
		for (int i = 0; i < n / 2; i++)
			a[i] += a[n - i - 1];
		n = n / 2;
		Recursion(a, n);
	}
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
		Recursion(a, n);
	}
	auto afterTime = std::chrono::steady_clock::now();
	double time = std::chrono::duration<double>(afterTime - beforeTime).count();
	cout << "Cache Recursion time=" << time << "seconds" << endl;
	return 0;
}