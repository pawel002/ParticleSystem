#ifndef SHADER_CLASS_H
#define SHADER_CLASS_H

#include <glad/glad.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cerrno>

#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtc/random.hpp>

std::string getFileContents(const char* filename);


class Shader {

	public:
		GLuint ID;
		Shader(const char* vertex, const char* frag);

		void _activate();
		void _delete();
		void setMat4(const std::string& name, const glm::mat4& mat);
		void setVec3(const std::string& name, const glm::vec3& value);

};

#endif // !SHADER_CLASS_H
