#ifndef PARTICLE_SIM_CLASS_H
#define PARTICLE_SIM_CLASS_H

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

#include "cameraClass.h"
#include "shaderClass.h"

class ParticleSim {

public:

	// simulation variables
	int particleCount;
	int typeCount;
	int blockSize;
	int linearBlockSize;
	float dragCoef;
	float rmin;
	float rmax;
	float deltaTime;
	float speedMultiplier;

	// window variables
	int width;
	int height;
	Camera camera;
	float lastX;
	float lastY;
	bool firstMouse;
	bool cursorInsideWindow;

	// imgui variables / objects
	int frames;
	float wireframeColor[3] = { 0.7, 0.7, 0.7 };
	float color[3 * 6] =
	{ 1.0, 0.0, 0.0,
	  0.0, 1.0, 0.0,
	  0.0, 0.0, 1.0,
	  0.0, 1.0, 1.0,
	  1.0, 0.0, 1.0,
	  1.0, 1.0, 0.0 };
	int newParticleCount;
	int newTypeCount;
	float newrmin;
	float newrmax;
	float newdragcoef;

	// pointers to data stored on cpu
	int* particleTypes;
	float* particleColors;
	float* particlePositions;
	float* particleVelocities;
	float* attractionMatrix;

	// pointers to data stored on gpu
	float* d_particlePositions;
	float* d_particleVelocities;
	float* d_constants;
	float* d_attractionMatrix;
	int* d_particleTypes;
	cudaGraphicsResource* cudaGR;
	size_t size;
	dim3 grid;
	dim3 threads;

	// openGL objects
	GLFWwindow* window = NULL;
	GLuint vertexArray;
	GLuint vertexBuffer;
	GLuint typeBuffer;

	// random number gnerator
	std::random_device rd;
	std::mt19937 e2 = std::mt19937(rd());

	// cube container
	bool renderCubeContainer;
	GLuint cubeBuffer;
	GLuint cubeWireframeColorBuffer;

	// space partition acceleration
	bool betterSpacePartition;
	int partitionDimension;
	int* d_spacePartitionMasses;
	float* d_spacePartitionPositions;

public:

	ParticleSim(int width, int height);

	int initGLFW(int width, int height);
	int initImGUI();

	void renderImGui(float fps);

	void setAttractionMatrix();

	void generateParticles();
	void mainLoop();
	void callDevice();

	void mouseCallback(double xposIn, double yposIn);
	void scrollCallback(double xoffset, double yoffset);
	void processInput(GLFWwindow* window);
	
	void enableCubeContainer();

	void enableBetterSpacePartition();
};

#endif // !PARTICLE_SIM_CLASS_H
