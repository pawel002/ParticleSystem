#ifndef KERNEL_H
#define KERNEL_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaGL.h"
#include "cuda_gl_interop.h"

__global__
void computeDrag(float* d_particleVelocities,
	float* d_constants);

__global__
void computeBounds(float* d_particlePositions,
	float* d_particleVelocities,
	float* d_constants,
	float dt);

__global__
void computeDisplacement(float* d_particlePositions,
	float* d_particleVelocities,
	float* d_constants);

__global__
void computeForces(
	float* d_particlePositions,
	float* d_particleVelocities,
	int* d_particleTypes,
	float* d_attractionMatrix,
	float* d_constants,
	float dt,
	int typeCount);

__global__
void computeForcesShared(
	float* d_particlePositions,
	float* d_particleVelocities,
	int* d_particleTypes,
	float* d_attractionMatrix,
	float* d_constants,
	float dt,
	int typeCount);

__global__
void computeSpacePartition(
	float* d_particlePositions,
	int* d_spacePartitionMasses,
	float* d_spacePartitionPositions,
	float* d_constants,
	int* d_particleTypes,
	int typeCount,
	int rmax,
	int partitionDimension);

__global__
void reduceSpacePartitionMassCenters(
	float* d_particlePositions,
	int* d_spacePartitionMasses,
	float* d_spacePartitionPositions,
	float* d_constants,
	int* d_particleTypes,
	int typeCount,
	int rmax,
	int partitionDimension);


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
	int partitionDimension);

__global__
void zeroSpacePartitionMassCenters(
	int* d_spacePartitionMasses,
	float* d_spacePartitionPositions,
	int partitionDimension,
	int typeCount);

#endif // !KERNEL_H