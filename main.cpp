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


#include "particleSimClass.cuh"

ParticleSim particleSim(1200, 1200);

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn){

    ImGui_ImplGlfw_CursorPosCallback(window, xposIn, yposIn);

    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (particleSim.firstMouse){ 

        particleSim.lastX = xpos;
        particleSim.lastY = ypos;
        particleSim.firstMouse = false;
    }

    float xoffset = xpos - particleSim.lastX;
    float yoffset = particleSim.lastY - ypos;

    particleSim.lastX = xpos;
    particleSim.lastY = ypos;

    if (particleSim.cursorInsideWindow)
        particleSim.camera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset){

    particleSim.camera.ProcessMouseScroll(static_cast<float>(yoffset));
}


int main() {
    
    glfwSetCursorPosCallback(particleSim.window, mouse_callback);
    glfwSetScrollCallback(particleSim.window, scroll_callback);
	particleSim.generateParticles();
    particleSim.enableCubeContainer();
    //particleSim.enableBetterSpacePartition();
	particleSim.mainLoop();

}