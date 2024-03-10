#include<iostream>
#include <chrono>
using namespace std;
void Circle(int* a, int n)
{
	for (int m = n; m > 1; m /= 2) // log(n)¸ö²½Öè
		for (int i = 0; i < m / 2; i++)
			a[i] = a[i * 2] + a[i * 2 + 1];
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
		Circle(a, n);
	}
	auto afterTime = std::chrono::steady_clock::now();
	double time = std::chrono::duration<double>(afterTime - beforeTime).count();
	cout << "Cache Circle time=" << time << "seconds" << endl;
	return 0;
}