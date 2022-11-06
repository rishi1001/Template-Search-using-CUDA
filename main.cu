#include <stdio.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <queue>

using namespace std;

__device__
double getValue(double *pre, int i, int j, int N) {
	return ((i >= 0 && j >= 0) ? pre[j * N + i] : 0);
}

__global__
void kernel(int N, int M, int n, int m, double th1, double th2, double avg, int *data, int *query, double *result, double *pre) {
    int corner_x = blockIdx.x * 32 + threadIdx.x;
	int corner_y = blockIdx.y * 32 + threadIdx.y;

	double gray = 0, x1, x2, y1, y2, cos, sin;

	if(blockIdx.z == 0) {
		cos = 0.707; sin = -0.707;
		x1 = corner_x; x2 = corner_x + (n - 1) * cos - (m - 1) * sin;
		y1 = corner_y + (n - 1) * sin; y2 = corner_y + (m - 1) * cos;
	} else if(blockIdx.z == 1) {
		cos = 1; sin = 0;
		x1 = corner_x; x2 = corner_x + n - 1;
		y1 = corner_y; y2 = corner_y + m - 1;
	} else {
		cos = 0.707; sin = 0.707;
		x1 = corner_x - (m - 1) * sin; x2 = corner_x + (n - 1) * cos;
		y1 = corner_y; y2 = corner_y + (n - 1) * sin + (m - 1) * cos;
	}

	if(floor(x1) < 0 || ceil(x2) >= N || floor(y1) < 0 || ceil(y2) >= M) {
		if(corner_y * N * 3 + corner_x * 3 + blockIdx.z < N*M*3) result[corner_y * N * 3 + corner_x * 3 + blockIdx.z] = INT_MAX;
		return;
	}

	gray = getValue(pre, floor(x2), floor(y2), N) - getValue(pre, floor(x2), ceil(y1) - 1, N) - getValue(pre, ceil(x1) - 1, floor(y2), N) + getValue(pre, ceil(x1) - 1, ceil(y1) - 1, N);
	gray /= ((floor(x2) - ceil(x1) + 1) * (floor(y2) - ceil(y1) + 1));

	if(abs(gray - avg) > th2) {
		result[corner_y * N * 3 + corner_x * 3 + blockIdx.z] = INT_MAX;
		return;
	}

	double rotate_x, rotate_y, scaled_x, scaled_y, interpolated, ans = 0;
	int query_data, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y, z00, z01, z10, z11;

	for(int j = 0; j < m; j++) {
		for(int i = 0; i < n; i++) {
			for(int k = 0; k < 3; k++) {
				rotate_x = corner_x + i*cos - j*sin;
				rotate_y = corner_y + i*sin + j*cos;

				p1_x = floor(rotate_x), p1_y = floor(rotate_y);
				p2_x = floor(rotate_x), p2_y = ceil(rotate_y);
				p3_x = ceil(rotate_x), p3_y = floor(rotate_y);
				p4_x = ceil(rotate_x), p4_y = ceil(rotate_y);

				scaled_x = rotate_x - p1_x;
				scaled_y = rotate_y - p1_y;

				z00 = data[p1_y * N * 3 + p1_x * 3 + k];
				z01 = data[p2_y * N * 3 + p2_x * 3 + k];
				z10 = data[p3_y * N * 3 + p3_x * 3 + k];
				z11 = data[p4_y * N * 3 + p4_x * 3 + k];

				query_data = query[j * n * 3 + i * 3 + k];

				interpolated = z00 * (1 - scaled_x) * (1 - scaled_y) + z10 * scaled_x * (1 - scaled_y) + z01 * (1 - scaled_x) * scaled_y + z11 * scaled_x * scaled_y;
				ans += (interpolated - query_data) * (interpolated - query_data);
				if(ans > th1 * th1 * (m*n*3)) {
					result[corner_y * N * 3 + corner_x * 3 + blockIdx.z] = INT_MAX;
					return;
				}
			}
		}
	}

	ans = sqrt(ans / (m*n*3));
	result[corner_y * N * 3 + corner_x * 3 + blockIdx.z] = ans;
}

int main(int argc, char *argv[]) {
	int angles[] = {-45, 0, 45};

    ifstream fs_data, fs_query;
    fs_data.open(argv[1], ios::in);
	fs_query.open(argv[2], ios::in);

	int N, M, n, m;
	fs_data >> M >> N; fs_query >> m >> n;

	double th1 = stof(argv[3]), th2 = stof(argv[4]);
	int output_n = stoi(argv[5]);
	
	int *data, *query, *d_data, *d_query;
	double *result, *d_result, *pre, *d_pre;
	data = (int *)malloc(N*M*3*sizeof(int));
	query = (int *)malloc(n*m*3*sizeof(int));
	result = (double *)malloc(N*M*3*sizeof(double));
	pre = (double *)malloc(N*M*sizeof(double));
	cudaMalloc(&d_data, N*M*3*sizeof(int));
	cudaMalloc(&d_query, n*m*3*sizeof(int));
	cudaMalloc(&d_result, N*M*3*sizeof(double));
	cudaMalloc(&d_pre, N*M*sizeof(double));

	for(int j = M - 1; j >= 0; j--) {
		for(int i = 0; i < N; i++) {
			for(int k = 0; k < 3; k++) {
				fs_data >> data[j*N*3 + i*3 + k];
			}
		}
	}

	for(int j = m - 1; j >= 0 ; j--) {
		for(int i = 0; i < n; i++) {
			for(int k = 0; k < 3 ; k++) {
				fs_query >> query[j*n*3 + i*3 + k];
			}
		}
	}

	cudaMemcpy(d_data, data, N*M*3*sizeof(int), cudaMemcpyDefault);
	cudaMemcpy(d_query, query, n*m*3*sizeof(int), cudaMemcpyDefault);

	double avg = 0;
	for(int j = 0; j < m; j++) {
		for(int i = 0; i < n; i++) {
			avg += (double) (query[j * n * 3 + i * 3] + query[j * n * 3 + i * 3 + 1] + query[j * n * 3 + i * 3 + 2]) / 3;
		}
	}
	avg /= (n * m);

	for(int j = 0; j < M; j++) {
		for(int i = 0; i < N; i++) {
			if(i == 0 && j == 0) {
				pre[j * N + i] = (double) (data[j * N * 3 + i * 3] + data[j * N * 3 + i * 3 + 1] + data[j * N * 3 + i * 3 + 2]) / 3;
			} else if(i == 0) {
				pre[j * N + i] = pre[(j - 1) * N + i] + (double) (data[j * N * 3 + i * 3] + data[j * N * 3 + i * 3 + 1] + data[j * N * 3 + i * 3 + 2]) / 3;
			} else if(j == 0) {
				pre[j * N + i] = pre[j * N + (i - 1)] + (double) (data[j * N * 3 + i * 3] + data[j * N * 3 + i * 3 + 1] + data[j * N * 3 + i * 3 + 2]) / 3;
			} else {
				pre[j * N + i] = pre[(j - 1) * N + i] + pre[j * N + (i - 1)] - pre[(j - 1) * N + (i - 1)] + (double) (data[j * N * 3 + i * 3] + data[j * N * 3 + i * 3 + 1] + data[j * N * 3 + i * 3 + 2]) / 3;
			}
		}
	}

	cudaMemcpy(d_pre, pre, N*M*sizeof(double), cudaMemcpyDefault);

	dim3 num_blocks((N + 31) / 32, (M + 31) / 32, 3);
	dim3 num_threads(32, 32);

	kernel<<<num_blocks, num_threads>>>(N, M, n, m, th1, th2, avg, d_data, d_query, d_result, d_pre);

	cudaMemcpy(result, d_result, N*M*3*sizeof(double), cudaMemcpyDefault);

	if(output_n > 1) {
		vector<tuple<double,int, int, int>> v;
		for(int j=0; j< M ; j++){
			for(int i=0; i< N; i++){
				for(int k=0; k<3 ; k++){
					tuple<double, int, int, int> n1(result[j*N*3 + i*3 + k], i, j, k);
					v.push_back(n1);
				}
			}
		}
		priority_queue<tuple<double, int, int, int>, vector<tuple<double, int, int, int>>, greater<tuple<double, int, int, int>>> pq(v.begin(), v.end()); 
		for(int i = 0; i < output_n; i++) {
			tuple<double, int, int, int> t = pq.top();
			pq.pop();
			if( get<0>(t) > th1){
				break;
			}
			cout << get<2>(t) << " " << get<1>(t) << " " << angles[get<3>(t)] << "\n";
			// cout << get<0>(t) << "\n";
		}
		free(data); free(query); free(result); cudaFree(d_data); cudaFree(d_query); cudaFree(d_result);
		return 0;
	}

	double min_val = (double) INT_MAX, I = -1, J = -1, K = -1;
	for(int j = 0; j < M; j++) {
		for(int i = 0; i < N; i++) {
			for(int k = 0; k < 3; k++) {
				if(result[j*N*3 + i*3 + k] < min_val) {
					min_val = result[j*N*3 + i*3 + k];
					I = i; J = j; K = k;
				}
			}
		}
	}

	free(data); free(query); free(result); cudaFree(d_data); cudaFree(d_query); cudaFree(d_result);

	if(min_val>th1){
		return 0;
	}
	cout << J << " " << I << " " << angles[static_cast<int>(K)] << "\n";
	// cout << min_val << "\n";
}