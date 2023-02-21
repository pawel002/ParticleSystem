#include <stdio.h>
#include <stdlib.h> 
#include <algorithm>
#include <random>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtc/random.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaGL.h"
#include "cuda_gl_interop.h"

#include "shaderClass.h"
#include "kernel.cuh"

using namespace std;

#define SQUARED_DRAG_FORCE true
#define FORCES_SHARED_MEMORY_SIZE 16

#define RED 0
#define GREEN 1
#define BLUE 2
#define INDIGO 3
#define YELLOW 4
#define ORANGE 5
#define VIOLET 6


__global__
void computeDrag(
	float* d_particleVelocities,
	float* d_constants){

	int size = (int) d_constants[3];

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;

	float d = 1.0f;

	if (SQUARED_DRAG_FORCE) {

		float vx = d_particleVelocities[3 * i];
		float vy = d_particleVelocities[3 * i + 1];
		float vz = d_particleVelocities[3 * i + 2];

		d = sqrt(vx * vx + vy * vy + vz * vz);
	}

	d_particleVelocities[3 * i] *= (1 - d_constants[0] * d);
	d_particleVelocities[3 * i + 1] *= (1 - d_constants[0] * d);
	d_particleVelocities[3 * i + 2] *= (1 - d_constants[0] * d);
}

__global__
void computeBounds(
	float* d_particlePositions,
	float* d_particleVelocities,
	float* d_constants,
	float dt) {

	int size = (int) d_constants[3];

	int i = 3 * (blockIdx.x * blockDim.x + threadIdx.x);
	if (i >= 3 * size)
		return;

	float x = d_particlePositions[i];
	float y = d_particlePositions[i + 1];
	float z = d_particlePositions[i + 2];

	if (x < -1 || x > 1) {

		d_particleVelocities[i] *= -1.0f;
		d_particlePositions[i] = x < 0 ? -0.99f : 0.99f;
	}

	if (y < -1 || y > 1) {

		d_particleVelocities[i + 1] *= -1.0f;
		d_particlePositions[i + 1] = y < 0 ? -0.99f : 0.99f;
	}

	if (z < -1 || z > 1) {

		d_particleVelocities[i + 2] *= -1.0f;
		d_particlePositions[i + 2] = z < 0 ? -0.99f : 0.99f;
	}
	
}

__global__
void computeDisplacement(
	float* d_particlePositions,
	float* d_particleVelocities,
	float* d_constants) {

	int size = (int) d_constants[3];

	int i = 3 * (blockIdx.x * blockDim.x + threadIdx.x);
	if (i >= 3 * size)
		return;

	d_particlePositions[i] += d_particleVelocities[i];
	d_particlePositions[i + 1] += d_particleVelocities[i + 1];
	d_particlePositions[i + 2] += d_particleVelocities[i + 2];
}

__global__
void computeForces(
	float* d_particlePositions,
	float* d_particleVelocities,
	int* d_particleTypes,
	float* d_attractionMatrix,
	float* d_constants,
	float dt,
	int typeCount) {

	int size = (int) d_constants[3];
	
	int i = 3 * (blockIdx.x * blockDim.x + threadIdx.x);
	int j = 3 * (blockIdx.y * blockDim.y + threadIdx.y);

	if (i >= 3 * size || j >= 3 * size || i == j)
		return;

	float dx = d_particlePositions[j] - d_particlePositions[i];
	float dy = d_particlePositions[j + 1] - d_particlePositions[i + 1];
	float dz = d_particlePositions[j + 2] - d_particlePositions[i + 2];

	if (dx > d_constants[2] || dy > d_constants[2] || dz > d_constants[2])
		return;

	float dist = sqrt(dx * dx + dy * dy + dz * dz);

	if (dist > d_constants[2])
		return;

	float scale;

	if (0.0f <= dist && dist <= d_constants[1])
		scale = dist / d_constants[1] - 1;
	else {

		if (d_constants[1] < dist && dist <= d_constants[2]) {

			float a = 2 / (d_constants[2] - d_constants[1]);
			float b = 1 - d_constants[2] * a;
			scale = 1 - abs(a * dist + b);

		}
		else
			scale = 0.0f;
	}

	int i_type = d_particleTypes[i / 3];
	int j_type = d_particleTypes[j / 3];
	float attractionFactor = d_attractionMatrix[i_type * typeCount + j_type];

	float dist_inv = 1 / dist;
	scale *= dt * 0.0001f * dist_inv * attractionFactor;
	float dvx = dx * scale;
	float dvy = dy * scale;
	float dvz = dz * scale;

	d_particleVelocities[i] += dvx;
	d_particleVelocities[i + 1] += dvy;
	d_particleVelocities[i + 2] += dvz;
}

__global__
void computeForcesShared(
	float* d_particlePositions,
	float* d_particleVelocities,
	int* d_particleTypes,
	float* d_attractionMatrix,
	float* d_constants,
	float dt,
	int typeCount) {

	int size = (int) d_constants[3];

	int i = 3 * (blockIdx.x * blockDim.x + threadIdx.x);
	int j = 3 * (blockIdx.y * blockDim.y + threadIdx.y);

	if (i >= 3 * size || j >= 3 * size || i == j)
		return;

	// size is equal to 3 times block size which for now is equal to 16
	extern __shared__ float P[24];
	extern __shared__ float Q[24];

	if (threadIdx.y == 0) {
		P[3*threadIdx.x + 0] = d_particlePositions[i];
		P[3*threadIdx.x + 1] = d_particlePositions[i + 1];
		P[3*threadIdx.x + 2] = d_particlePositions[i + 2];
	}

	if (threadIdx.x == 0) {
		Q[3*threadIdx.y + 0] = d_particlePositions[j];
		Q[3*threadIdx.y + 1] = d_particlePositions[j + 1];
		Q[3*threadIdx.y + 2] = d_particlePositions[j + 2];
	}

	int a = i;
	int b = j;
	i = 3 * threadIdx.x;
	j = 3 * threadIdx.y;

	__syncthreads();

	float dx = Q[j] - P[i];
	float dy = Q[j + 1] - P[i + 1];
	float dz = Q[j + 2] - P[i + 2];

	if (dx > d_constants[2] || dy > d_constants[2] || dz > d_constants[2])
		return;

	float dist = sqrt(dx * dx + dy * dy + dz * dz);

	if (dist > d_constants[2])
		return;

	float scale;

	if (0.0f <= dist && dist <= d_constants[1])
		scale = dist / d_constants[1] - 1;
	else {

		if (d_constants[1] < dist && dist <= d_constants[2]) {

			float a = 2 / (d_constants[2] - d_constants[1]);
			float b = 1 - d_constants[2] * a;
			scale = 1 - abs(a * dist + b);

		}
		else
			scale = 0.0f;
	}

	int i_type = d_particleTypes[a / 3];
	int j_type = d_particleTypes[b / 3];
	float attractionFactor = d_attractionMatrix[i_type * typeCount + j_type];

	float dist_inv = 1 / dist;
	scale *= dt * 0.0001f * dist_inv * attractionFactor;
	float dvx = dx * scale;
	float dvy = dy * scale;
	float dvz = dz * scale;

	d_particleVelocities[a] += dvx;
	d_particleVelocities[a + 1] += dvy;
	d_particleVelocities[a + 2] += dvz;
}


__global__
void computeSpacePartition(
	float* d_particlePositions,
	int* d_spacePartitionMasses,
	float* d_spacePartitionPositions,
	float* d_constants,
	int* d_particleTypes,
	int typeCount,
	int rmax,
	int partitionDimension) {

	int size = (int)d_constants[3];

	int i = 3 * (blockIdx.x * blockDim.x + threadIdx.x);
	if (i >= 3 * size)
		return;

	float a = d_particlePositions[i];
	float b = d_particlePositions[i + 1];
	float c = d_particlePositions[i + 2];

	float inv = 1 / rmax;
	int x = floor((a + 1) * inv);
	int y = floor((b + 1) * inv);
	int z = floor((c + 1) * inv);
	
	int idx = z * partitionDimension * partitionDimension + y * partitionDimension + x;
	idx *= typeCount;

	int type = d_particleTypes[blockIdx.x * blockDim.x + threadIdx.x];

	d_spacePartitionMasses[idx + type] += 1;

	d_spacePartitionPositions[3 * idx] += a;
	d_spacePartitionPositions[3 * idx] += b;
	d_spacePartitionPositions[3 * idx] += c;
}

// needs to be called for every cell in the 3d space
// 3d architecture can be usefull
__global__
void reduceSpacePartitionMassCenters(
	float* d_particlePositions,
	int* d_spacePartitionMasses,
	float* d_spacePartitionPositions,
	float* d_constants,
	int* d_particleTypes,
	int typeCount,
	int rmax,
	int partitionDimension) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (z >= partitionDimension || y >= partitionDimension || x >= partitionDimension)
		return;

	int idx = z * partitionDimension * partitionDimension + y * partitionDimension + x;
	idx *= typeCount;
	for (int i = 0; i < typeCount; i++) {
		d_spacePartitionPositions[3 * idx + 3 * i + 0] /= (float) d_spacePartitionMasses[idx + i];
		d_spacePartitionPositions[3 * idx + 3 * i + 1] /= (float) d_spacePartitionMasses[idx + i];
		d_spacePartitionPositions[3 * idx + 3 * i + 2] /= (float) d_spacePartitionMasses[idx + i];
	}
}

__global__
void computeSpacePartitionForces(
	float* d_particlePositions,
	float* d_particleVelocities,
	int* d_spacePartitionMasses,
	float* d_spacePartitionPositions,
	float* d_constants,
	int* d_particleTypes,
	float* d_attractionMatrix,
	int typeCount,
	int rmax,
	float dt,
	int partitionDimension) {

	int size = (int)d_constants[3];

	int i = 3 * (blockIdx.x * blockDim.x + threadIdx.x);
	if (i >= 3 * size)
		return;

	float a = d_particlePositions[i];
	float b = d_particlePositions[i + 1];
	float c = d_particlePositions[i + 2];

	float inv = 1 / rmax;
	int x = floor((a + 1) * inv);
	int y = floor((b + 1) * inv);
	int z = floor((c + 1) * inv);

	int idx = z * partitionDimension * partitionDimension + y * partitionDimension + x;
	idx *= typeCount;

	int type = d_particleTypes[blockIdx.x * blockDim.x + threadIdx.x];
	int p, q, r;

	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			for (int k = -1; k <= 1; k++) {
				
				p = x + i;
				q = y + j;
				r = z + k;

				if (p < 0 || p > partitionDimension)
					continue;

				if (q < 0 || q > partitionDimension)
					continue;

				if (r < 0 || r > partitionDimension)
					continue;

				// get index of a cell 

				int forceIndex = r * partitionDimension * partitionDimension + q * partitionDimension + p;
				for (int w = 0; w < typeCount; w++) {

					float dx = d_spacePartitionPositions[3 * forceIndex + 3 * w + 0] - a;
					float dy = d_spacePartitionPositions[3 * forceIndex + 3 * w + 1] - b;
					float dz = d_spacePartitionPositions[3 * forceIndex + 3 * w + 2] - c;

					if (dx > d_constants[2] || dy > d_constants[2] || dz > d_constants[2])
						return;

					float dist = sqrt(dx * dx + dy * dy + dz * dz);

					if (dist > d_constants[2])
						return;

					float scale;

					if (0.0f <= dist && dist <= d_constants[1])
						scale = dist / d_constants[1] - 1;
					else {

						if (d_constants[1] < dist && dist <= d_constants[2]) {

							float a = 2 / (d_constants[2] - d_constants[1]);
							float b = 1 - d_constants[2] * a;
							scale = 1 - abs(a * dist + b);

						}
						else
							scale = 0.0f;
					}

					float attractionFactor = d_attractionMatrix[type * typeCount + w];

					float dist_inv = 1 / dist;
					scale *= dt * 0.0001f * dist_inv * attractionFactor * d_spacePartitionMasses[forceIndex + w];
					float dvx = dx * scale;
					float dvy = dy * scale;
					float dvz = dz * scale;

					d_particleVelocities[i] += dvx;
					d_particleVelocities[i + 1] += dvy;
					d_particleVelocities[i + 2] += dvz;
				}
			}
		}
	}
}

__global__
void zeroSpacePartitionMassCenters(
	int* d_spacePartitionMasses,
	float* d_spacePartitionPositions,
	int partitionDimension,
	int typeCount) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (z >= partitionDimension || y >= partitionDimension || x >= partitionDimension)
		return;

	int idx = z * partitionDimension * partitionDimension + y * partitionDimension + x;
	idx *= typeCount;
	for (int i = 0; i < typeCount; i++) {
		d_spacePartitionPositions[3 * idx + 3 * i + 0] = 0;
		d_spacePartitionPositions[3 * idx + 3 * i + 1] = 0;
		d_spacePartitionPositions[3 * idx + 3 * i + 2] = 0;
		d_spacePartitionMasses[idx + i] = 0;
	}
}



