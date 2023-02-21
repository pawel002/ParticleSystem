#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/random.hpp>

#include <iostream>
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <vector>
#include <functional>
#include <future>
#include <chrono>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaGL.h"
#include "cuda_gl_interop.h"

#include "shaderClass.h"
#include "particleSimClass.cuh"
#include "kernel.cuh"
#include "cameraClass.h"


#define RED 0
#define GREEN 1
#define BLUE 2
#define INDIGO 3
#define YELLOW 4
#define ORANGE 5
#define VIOLET 6

#define M_PI 3.14159265358979323846

ParticleSim::ParticleSim(int width, int height) {

	this->width = width;
	this->height = height;
	firstMouse = true;

	// initialize opengl + glfw
	initGLFW(width, height);

	// initialize IMGUI
	initImGUI();

	// for now defining parameters happens here
	particleCount = 4000;
	typeCount = 3;
	dragCoef = 0.4f;
	rmin = 0.1f;
	rmax = 0.3f;

	blockSize = 8;
	linearBlockSize = 512;

	speedMultiplier = 1.0;

	srand(100);
	e2.seed(101);

	setAttractionMatrix();
}

void ParticleSim::setAttractionMatrix() {

	attractionMatrix = (float*)calloc(typeCount * typeCount, sizeof(float));
	std::uniform_real_distribution<> uniform(-0.99f, 0.99f);

	for (int i = 0; i < typeCount * typeCount; i++) {
		attractionMatrix[i] = (float)uniform(e2);
	}
}


int ParticleSim::initGLFW(int width, int height) {

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	lastX = width / 2.0f;
	lastY = height / 2.0f;

	window = glfwCreateWindow(width, height, "ParticleSimulator", NULL, NULL);
	if (!window) {
		std::cout << "Failed to create a window." << "\n";
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	gladLoadGL();
	glViewport(0, 0, width, height);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glfwSwapBuffers(window);

	glEnable(GL_PROGRAM_POINT_SIZE);

	// depth test
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	// vertex array
	glGenVertexArrays(1, &vertexArray);
	glBindVertexArray(vertexArray);

	// capture mouse
	cursorInsideWindow = false;

	return 1;
}

int ParticleSim::initImGUI() {

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	(void) io;
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 460 core");
	return 1;
}

void ParticleSim::renderImGui(float fps) {

	ImGui::Begin("Options");
	ImGui::Text(("FPS " + std::to_string(fps)).c_str());

	// color editors
	ImGui::ColorEdit3("Wireframe Color", wireframeColor);
	ImGui::ColorEdit3("Color1", color1);
	ImGui::ColorEdit3("Color2", color2);
	ImGui::ColorEdit3("Color3", color3);
	ImGui::ColorEdit3("Color4", color4);
	ImGui::ColorEdit3("Color5", color5);
	ImGui::ColorEdit3("Color6", color6);


	ImGui::End();
}


void ParticleSim::generateParticles() {

	std::uniform_real_distribution<> uniform(-0.99f, 0.99f);

	// allocate memory on cpu for generation purposes
	particleTypes = (int*)calloc(particleCount, sizeof(int));
	particlePositions = (float*)calloc(3 * particleCount, sizeof(float));
	particleVelocities = (float*)calloc(3 * particleCount, sizeof(float));

	// generate particle types, positions and colors
	for (int i = 0; i < particleCount; i++) {
		particlePositions[3 * i] = (float) uniform(e2);
		particlePositions[3 * i + 1] = (float) uniform(e2);
		particlePositions[3 * i + 2] = (float) uniform(e2);
		particleTypes[i] = rand() % typeCount;
	}

	// particle type buffer
	glGenBuffers(1, &typeBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, typeBuffer);
	glBufferData(GL_ARRAY_BUFFER, particleCount * sizeof(int), particleTypes, GL_STATIC_READ);

	// particle vertices buffer
	glGenBuffers(1, &vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, 3 * particleCount * sizeof(float), particlePositions, GL_DYNAMIC_DRAW);
	cudaGraphicsGLRegisterBuffer(&cudaGR, vertexBuffer, 0);

	// cuda malloc
	cudaMalloc(&d_particleVelocities, 3 * particleCount * sizeof(float));
	cudaMalloc(&d_particleTypes, particleCount * sizeof(int));
	cudaMalloc(&d_constants, 4 * sizeof(float));
	cudaMalloc(&d_attractionMatrix, typeCount * typeCount * sizeof(float));

	float constants[] = {dragCoef, rmin, rmax, particleCount};
	cudaMemcpy(d_particleVelocities, particleVelocities, 3 * particleCount * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_particleTypes, particleTypes, particleCount * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_constants, constants, 4 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_attractionMatrix, attractionMatrix, typeCount * typeCount * sizeof(float), cudaMemcpyHostToDevice);

	cudaGraphicsMapResources(1, &cudaGR, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_particlePositions, &size, cudaGR);

	int gridSize = (int)ceil(particleCount / blockSize);
	grid = dim3(gridSize, gridSize);
	threads = dim3(blockSize, blockSize);
}


void ParticleSim::mainLoop() {

	// create class that stores shaders
	Shader shaderProgram("default.vert", "default.frag");
	shaderProgram._activate();
	
	// frames check
	double lastTime = glfwGetTime(), timer = lastTime, lastFrame = lastTime;
	double nowTime = 0;
	frames = 0;
	float fps = 0;
	

	// camera
	camera = Camera();

	while (!glfwWindowShouldClose(window)) {

		processInput(window);

		// time
		nowTime = glfwGetTime();
		deltaTime = (float) speedMultiplier * (nowTime - lastTime);
		lastTime = nowTime;

		// get projection matrix
		glm::mat4 projection = glm::perspective<float>(glm::radians(60.0), (float)width / (float)height, 0.1f, 100.0f);
		shaderProgram.setMat4("projection", projection);
		glm::mat4 view = camera.GetViewMatrix();
		shaderProgram.setMat4("view", view);

		shaderProgram.setVec3("cameraEye", camera.Position);

		shaderProgram.setVec3("wireframeColor", glm::vec3(wireframeColor[0], wireframeColor[1], wireframeColor[2]));
		shaderProgram.setVec3("color1", glm::vec3(color1[0], color1[1], color1[2]));
		shaderProgram.setVec3("color2", glm::vec3(color2[0], color2[1], color2[2]));
		shaderProgram.setVec3("color3", glm::vec3(color3[0], color3[1], color3[2]));
		shaderProgram.setVec3("color4", glm::vec3(color4[0], color4[1], color4[2]));
		shaderProgram.setVec3("color5", glm::vec3(color5[0], color5[1], color5[2]));
		shaderProgram.setVec3("color6", glm::vec3(color6[0], color6[1], color6[2]));


		// clear
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// imgui newframe
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		
		// computes one step of simulation on the gpu
		callDevice();

		// draw particles
		glBindVertexArray(vertexArray);

		glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, typeBuffer);
		glVertexAttribIPointer(1, 1, GL_INT, sizeof(int), (void*)0);
		glEnableVertexAttribArray(1);

		glDrawArrays(GL_POINTS, 0, particleCount);

		// draw background cube
		if (renderCubeContainer) {

			glBindBuffer(GL_ARRAY_BUFFER, cubeBuffer);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);

			glBindBuffer(GL_ARRAY_BUFFER, cubeWireframeColorBuffer);
			glVertexAttribIPointer(1, 1, GL_INT, sizeof(int), (void*)0);
			glEnableVertexAttribArray(1);

			glDrawArrays(GL_TRIANGLES, 0, 36);
		}

		frames++;
		if (glfwGetTime() - lastFrame > 1.0f) {
			fps = (float)frames / (glfwGetTime() - lastFrame);
			lastFrame = glfwGetTime();
			frames = 0;
		}

		// draw imgui
		if(!cursorInsideWindow)
			renderImGui(fps);

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		// frames every 60 frames

		glfwSwapBuffers(window);
		glfwPollEvents();

	}
}

void ParticleSim::callDevice() {

	int dim3blocksize = 8;

	int linearGridSize = ceil(particleCount / linearBlockSize);

	int d3s = ceil(partitionDimension / dim3blocksize);
	dim3 dim3grid = dim3(d3s, d3s, d3s);
	dim3 dim3block = dim3(dim3blocksize, dim3blocksize, dim3blocksize);

	// map resources
	cudaGraphicsMapResources(1, &cudaGR, 0);

	// compute forces
	computeDrag <<< linearGridSize, linearBlockSize >>> (d_particleVelocities, d_constants);

	if (betterSpacePartition) {

		computeSpacePartition<<< linearGridSize, linearBlockSize >>> (d_particlePositions, d_spacePartitionMasses,d_spacePartitionPositions,d_constants,d_particleTypes,typeCount,rmax,partitionDimension);
		reduceSpacePartitionMassCenters << < dim3grid, dim3block >> > (d_particlePositions, d_spacePartitionMasses, d_spacePartitionPositions, d_constants, d_particleTypes, typeCount, rmax, partitionDimension);
		computeSpacePartitionForces << < linearGridSize, linearBlockSize >> > (d_particlePositions, d_particleVelocities, d_spacePartitionMasses, d_spacePartitionPositions, d_constants, d_particleTypes, d_attractionMatrix, typeCount, rmax, deltaTime, partitionDimension);
		zeroSpacePartitionMassCenters << < dim3grid, dim3block >>> (d_spacePartitionMasses, d_spacePartitionPositions, partitionDimension, typeCount);

	}
	else {

		computeForces << < grid, threads >> > (d_particlePositions, d_particleVelocities, d_particleTypes, d_attractionMatrix, d_constants, deltaTime, typeCount);

	}

	computeDisplacement <<< linearGridSize, linearBlockSize >>> (d_particlePositions, d_particleVelocities, d_constants);
	computeBounds <<< linearGridSize, linearBlockSize >>> (d_particlePositions, d_particleVelocities, d_constants, deltaTime);

	// unmap resources so they can be used by opengl
	cudaGraphicsUnmapResources(1, &cudaGR, 0);

}


void ParticleSim::mouseCallback(double xposIn, double yposIn) {

	float xpos = static_cast<float>(xposIn);
	float ypos = static_cast<float>(yposIn);

	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos;

	lastX = xpos;
	lastY = ypos;

	camera.ProcessMouseMovement(xoffset, yoffset);
}

void ParticleSim::scrollCallback(double xoffset, double yoffset) {

	camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

void ParticleSim::processInput(GLFWwindow* window) {

	if (cursorInsideWindow) {
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
			camera.ProcessKeyboard(FORWARD, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
			camera.ProcessKeyboard(BACKWARD, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
			camera.ProcessKeyboard(LEFT, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
			camera.ProcessKeyboard(RIGHT, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
			camera.ProcessKeyboard(UP, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
			camera.ProcessKeyboard(DOWN, deltaTime);
	}

	if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
		cursorInsideWindow = true;
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		cursorInsideWindow = false;
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}
	
}


void ParticleSim::enableCubeContainer() {

	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glFrontFace(GL_CW);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	renderCubeContainer = true;
	float cubeVertices[] = {
		    // Back face
		   -1.0f, -1.0f, -1.0f,// Bottom-left
			1.0f,  1.0f, -1.0f,// top-right
			1.0f, -1.0f, -1.0f,// bottom-right         
			1.0f,  1.0f, -1.0f,// top-right
		   -1.0f, -1.0f, -1.0f,// bottom-left
		   -1.0f,  1.0f, -1.0f,// top-left
		   // Front face	
		   -1.0f, -1.0f,  1.0f,// bottom-left
			1.0f, -1.0f,  1.0f,// bottom-right
			1.0f,  1.0f,  1.0f,// top-right
			1.0f,  1.0f,  1.0f,// top-right
		   -1.0f,  1.0f,  1.0f,// top-left
		   -1.0f, -1.0f,  1.0f,// bottom-left
		   // Left face		
		   -1.0f,  1.0f,  1.0f,// top-right
		   -1.0f,  1.0f, -1.0f,// top-left
		   -1.0f, -1.0f, -1.0f,// bottom-left
		   -1.0f, -1.0f, -1.0f,// bottom-left
		   -1.0f, -1.0f,  1.0f,// bottom-right
		   -1.0f,  1.0f,  1.0f,// top-right
		   // Right face	
			1.0f,  1.0f,  1.0f,// top-left
			1.0f, -1.0f, -1.0f,// bottom-right
			1.0f,  1.0f, -1.0f,// top-right         
			1.0f, -1.0f, -1.0f,// bottom-right
			1.0f,  1.0f,  1.0f,// top-left
			1.0f, -1.0f,  1.0f,// bottom-left     
			// Bottom face	
			-1.0f, -1.0f, -1.0f, // top-right
			 1.0f, -1.0f, -1.0f, // top-left
			 1.0f, -1.0f,  1.0f, // bottom-left
			 1.0f, -1.0f,  1.0f, // bottom-left
			-1.0f, -1.0f,  1.0f, // bottom-right
			-1.0f, -1.0f, -1.0f, // top-right
			// Top face		   
			-1.0f,  1.0f, -1.0f, // top-left
			 1.0f,  1.0f,  1.0f, // bottom-right
			 1.0f,  1.0f, -1.0f, // top-right     
			 1.0f,  1.0f,  1.0f, // bottom-right
			-1.0f,  1.0f, -1.0f, // top-left
			-1.0f,  1.0f,  1.0f // bottom-left            
	};

	glGenBuffers(1, &cubeBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, cubeBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);

	int wireframeColor[36];
	std::fill_n(wireframeColor, 36, 21);

	glGenBuffers(1, &cubeWireframeColorBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, cubeWireframeColorBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(wireframeColor), wireframeColor, GL_STATIC_DRAW);

}

void ParticleSim::enableBetterSpacePartition() {

	betterSpacePartition = true;
	partitionDimension = ceil(2.0f / rmax);
	cudaMalloc(&d_spacePartitionMasses, typeCount * partitionDimension * partitionDimension * partitionDimension * sizeof(int));
	cudaMalloc(&d_spacePartitionPositions, 3 * typeCount * partitionDimension * partitionDimension * partitionDimension * sizeof(float));
}