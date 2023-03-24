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
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <vector>
#include <functional>
#include <future>
#include <chrono>
#include <math.h>
#include <format>
#include <string_view>

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

	cudaDeviceProp prop;
	cudaError_t error = cudaGetDeviceProperties(&prop, 0);
	if (error != cudaSuccess) {
		std::cerr << "cudaGetDeviceProperties() failed: " << cudaGetErrorString(error) << std::endl;
	}
	std::cout << "Device name: " << prop.name << std::endl;

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
	rmin = 0.05f;
	rmax = 0.7f;

	blockSize = 8;
	linearBlockSize = 512;

	speedMultiplier = 1.0;

	srand(100);
	e2.seed(101);

	newParticleCount = particleCount;
	newTypeCount = typeCount;
	newrmin = rmin;
	newrmax = rmax;
	newdragcoef = dragCoef;

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

void addCenteredText(ImVec2 center, std::string text, ImDrawList* drawList) {

	auto textSize = ImGui::CalcTextSize(text.c_str());
	drawList->AddText(center - ImVec2(0.5f * textSize.x, 0.5f * textSize.y), ImColor(1.0f, 1.0f, 1.0f, 1.0f), text.c_str());
}

void ParticleSim::renderImGui(float fps) {

	ImGui::Begin("Options");

	ImDrawList* drawList = ImGui::GetWindowDrawList();
	ImVec2 p = ImGui::GetCursorScreenPos();
	float matrixYoffset = 270.0f + 25.0f * typeCount;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	std::string propName(prop.name);

	ImGui::Text(("Press C to exit Setting, RUNNIG ON DEVICE: " + propName).c_str());
	ImGui::Text(("FPS " + std::to_string(fps)).c_str());

	ImGui::InputInt("Number of particles", &newParticleCount, 100);
	ImGui::InputInt("Number of particle types", &newTypeCount, 1);
	ImGui::InputFloat("rmin", &newrmin, 0.05f);
	ImGui::InputFloat("rmax", &newrmax, 0.05f);
	ImGui::InputFloat("Drag coefficient", &newdragcoef, 0.05f);

	// color editors
	ImGui::ColorEdit3("Wireframe Color", wireframeColor);
	if (typeCount >= 1) ImGui::ColorEdit3("Color1", &color[0]);
	if (typeCount >= 2) ImGui::ColorEdit3("Color2", &color[3]);
	if (typeCount >= 3) ImGui::ColorEdit3("Color3", &color[6]);
	if (typeCount >= 4) ImGui::ColorEdit3("Color4", &color[9]);
	if (typeCount >= 5) ImGui::ColorEdit3("Color5", &color[12]);
	if (typeCount >= 6) ImGui::ColorEdit3("Color6", &color[15]);
	
	ImVec2 windowPosition = ImGui::GetWindowPos();
	ImVec2 windowSize = ImGui::GetWindowSize();
	float w = std::min(windowSize.x / (1.25f * typeCount), 50.0f);

	// draw color labels
	for (int i = 0; i < typeCount; i++) {

		drawList->AddRectFilled(ImVec2(i * w + w + 10.0, matrixYoffset) + windowPosition,
			ImVec2((i + 1) * w + w + 10.0, w + matrixYoffset) + windowPosition,
			ImColor(color[3 * i + 0], color[3 * i + 1], color[3 * i + 2], 1.0));

		drawList->AddRectFilled(ImVec2(10, i * w + matrixYoffset + w) + windowPosition,
			ImVec2(w + 10.0, (i + 1) * w + matrixYoffset + w) + windowPosition,
			ImColor(color[3 * i + 0], color[3 * i + 1], color[3 * i + 2], 1.0));
	}

	// create a attraction matrix 
	for (int i = 0; i < typeCount; i++) {
		for (int j = 0; j < typeCount; j++) {
			float val = (attractionMatrix[i * typeCount + j] + 1.0f) / 2.0f;
			ImVec2 upperLeft = ImVec2(i * w + 10.0 + w, j * w + matrixYoffset + w) + windowPosition;
			ImVec2 lowerRight = ImVec2((i + 1) * w + 10.0 + w, (j + 1) * w + matrixYoffset + w) + windowPosition;

			drawList->AddRectFilled(upperLeft, lowerRight, 
				ImColor(std::min(std::max(1.0f - val, 0.0f), 1.0f) , std::min(std::max(val, 0.0f), 1.0f), 0.0, 1.0));

			// handle scrolling
			if (ImGui::IsMouseHoveringRect(upperLeft, lowerRight)) {
				float scroll = ImGui::GetIO().MouseWheel / 10.0;
				attractionMatrix[i * typeCount + j] += scroll;
			}

			std::stringstream stream;
			stream << std::fixed << std::setprecision(2) << attractionMatrix[i * typeCount + j];
			std::string s = stream.str();

			addCenteredText(ImVec2((i + 0.5) * w + 10.0 + w, (j + 0.5) * w + matrixYoffset + w) + windowPosition, s, drawList);
		}
	}

	if (ImGui::Button("Reset Attraction Matrix")) {
		for (int i = 0; i < typeCount; i++) {
			for (int j = 0; j < typeCount; j++) {
				attractionMatrix[i * typeCount + j] = 0.0f;
			}
		}
	}

	if (ImGui::Button("Apply Changes")) {

		std::uniform_real_distribution<> uniform(-0.99f, 0.99f);

		// handle the change of the number of particles
		if (newParticleCount != particleCount) {

			// temp vectors
			float* tempParticlePositions = (float*) malloc(3 * particleCount * sizeof(float));
			float* tempParticleVelocities = (float*) malloc(3 * particleCount * sizeof(float));
			int* tempParticleTypes = (int*) calloc(3 * particleCount, sizeof(int));
			
			// copy infarmation from device memory
			cudaMemcpy(tempParticlePositions, d_particlePositions, 3 * particleCount * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(tempParticleVelocities, d_particleVelocities, 3 * particleCount * sizeof(float), cudaMemcpyDeviceToHost);

			memcpy(tempParticleTypes, particleTypes, particleCount * sizeof(int));

			// create new area for memory
			free(particlePositions);
			free(particleVelocities);
			free(particleTypes);

			particlePositions = (float*) calloc(3 * newParticleCount, sizeof(float));
			particleVelocities = (float*) calloc(3 * newParticleCount, sizeof(float));
			particleTypes = (int*) calloc(3 * newParticleCount, sizeof(int));

			memcpy(particlePositions, tempParticlePositions, 3 * std::min(newParticleCount, particleCount) * sizeof(float));
			memcpy(particleVelocities, tempParticleVelocities, 3 * std::min(newParticleCount, particleCount) * sizeof(float));
			memcpy(particleTypes, tempParticleTypes, std::min(newParticleCount, particleCount) * sizeof(int));

			// free cuda objects
			cudaFree(d_particleVelocities);
			cudaFree(d_particleTypes);

			// if newPC > currPC generate new positions/velocities/types
			if (newParticleCount > particleCount) {

				for (int i = 0; i < newParticleCount - particleCount; i++) {

					particlePositions[3 * particleCount + 3 * i + 0] = (float) uniform(e2);
					particlePositions[3 * particleCount + 3 * i + 1] = (float) uniform(e2);
					particlePositions[3 * particleCount + 3 * i + 2] = (float) uniform(e2);

					particleVelocities[3 * particleCount + 3 * i + 0] = 0.0f;
					particleVelocities[3 * particleCount + 3 * i + 1] = 0.0f;
					particleVelocities[3 * particleCount + 3 * i + 2] = 0.0f;

					particleTypes[particleCount + i] = rand() % typeCount;

				}
			}

			// allocate velocities and types - CUDA

			cudaMalloc(&d_particleVelocities, 3 * newParticleCount * sizeof(float));
			cudaMalloc(&d_particleTypes, newParticleCount * sizeof(int));

			cudaMemcpy(d_particleVelocities, particleVelocities, 3 * newParticleCount * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_particleTypes, particleTypes, newParticleCount * sizeof(int), cudaMemcpyHostToDevice);

			// allocate positions and types - openGL

			glBindBuffer(GL_ARRAY_BUFFER, typeBuffer);
			glBufferData(GL_ARRAY_BUFFER, newParticleCount * sizeof(int), particleTypes, GL_STATIC_READ);

			glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
			glBufferData(GL_ARRAY_BUFFER, 3 * newParticleCount * sizeof(float), particlePositions, GL_DYNAMIC_DRAW);
			cudaGraphicsGLRegisterBuffer(&cudaGR, vertexBuffer, 0);

			cudaGraphicsMapResources(1, &cudaGR, 0);
			cudaGraphicsResourceGetMappedPointer((void**)&d_particlePositions, &size, cudaGR);

			// free temps
			free(tempParticlePositions);
			free(tempParticleVelocities);
			free(tempParticleTypes);

			particleCount = newParticleCount;

		}

		// handle change of the numer of types
		if (newTypeCount != typeCount) {
			
			if (newTypeCount < typeCount) {

				for (int i = 0; i < particleCount; i++) {

					if (particleTypes[i] >= newTypeCount) {

						particleTypes[i] = rand() % newTypeCount;
					}
				}
			}

			if (newTypeCount > typeCount) {

				for (int i = 0; i < particleCount; i++) {

					int newType = rand() % newTypeCount;

					if (newType >= typeCount) {
						particleTypes[i] = newType;
					}
				}
			}

			cudaMemcpy(d_particleTypes, particleTypes, particleCount * sizeof(int), cudaMemcpyHostToDevice);

			float* newAttractionMatrix = (float*) calloc(newTypeCount * newTypeCount, sizeof(float));

			for (int i = 0; i < newTypeCount; i++) {
				for (int j = 0; j < newTypeCount; j++) {

					if (i < typeCount && j < typeCount) {
						newAttractionMatrix[i * newTypeCount + j] = attractionMatrix[i * typeCount + j];
					}
					else {
						newAttractionMatrix[i * newTypeCount + j] = (float) uniform(e2);
					}
				}
			}

			typeCount = newTypeCount;
			attractionMatrix = (float *) calloc(newTypeCount * newTypeCount, sizeof(float));
			for (int i = 0; i < newTypeCount * newTypeCount; i++) attractionMatrix[i] = newAttractionMatrix[i];
			free(newAttractionMatrix);

			glBindBuffer(GL_ARRAY_BUFFER, typeBuffer);
			glBufferData(GL_ARRAY_BUFFER, particleCount * sizeof(int), particleTypes, GL_STATIC_READ);
		}

		cudaFree(d_attractionMatrix);
		cudaMalloc(&d_attractionMatrix, newTypeCount * newTypeCount * sizeof(float));
		cudaMemcpy(d_attractionMatrix, attractionMatrix, newTypeCount * newTypeCount * sizeof(float), cudaMemcpyHostToDevice);

		// handle change of the minimum radius
		if (newrmin != rmin) {
			rmin = newrmin;
		}

		// handle change of the maximum radius
		if (newrmax != rmax) {
			rmax = newrmax;
		}

		// handle change of the drag coefficent
		if (newdragcoef != dragCoef) {
			dragCoef = newdragcoef;
		}

		float constants[] = { dragCoef, rmin, rmax, particleCount };
		cudaMemcpy(d_constants, constants, 4 * sizeof(float), cudaMemcpyHostToDevice);

		std::cout << "Applied changes\n";
	}
	
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
}


void ParticleSim::mainLoop() {

	// create class that stores shaders
	Shader shaderProgram("default.vert", "default.frag");
	shaderProgram._activate();
	
	// frames check
	float lastTime = glfwGetTime(), timer = lastTime, lastFrame = lastTime;
	float nowTime = 0;
	frames = 0;
	float fps = 0;
	

	// camera
	camera = Camera(glm::vec3(1.14115f, 1.08318f, 2.37937f));

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
		shaderProgram.setVec3("color1", glm::vec3(color[0], color[1], color[2]));
		shaderProgram.setVec3("color2", glm::vec3(color[3], color[4], color[5]));
		shaderProgram.setVec3("color3", glm::vec3(color[6], color[7], color[8]));
		shaderProgram.setVec3("color4", glm::vec3(color[9], color[10], color[11]));
		shaderProgram.setVec3("color5", glm::vec3(color[12], color[13], color[14]));
		shaderProgram.setVec3("color6", glm::vec3(color[15], color[16], color[17]));

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

	int linearGridSize = ceil(particleCount / linearBlockSize) + 1;

	int gridSize = (int)ceil(particleCount / blockSize) + 1;
	grid = dim3(gridSize, gridSize);
	threads = dim3(blockSize, blockSize);

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