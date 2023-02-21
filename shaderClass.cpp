#include "shaderClass.h"

std::string getFileContents(const char* filename) {

	std::ifstream in(filename, std::ios::binary);
	if (in) {
		std::string contents;
		in.seekg(0, std::ios::end);
		contents.resize(in.tellg());
		in.seekg(0, std::ios::beg);
		in.read(&contents[0], contents.size());
		in.close();
		return (contents);
	}

	throw(errno);
}

Shader::Shader(const char* vert, const char* frag) {
	std::string vertexCode = getFileContents(vert);
	std::string fragmentCode = getFileContents(frag);

	const char* vertexSource = vertexCode.c_str();
	const char* fragmentSource = fragmentCode.c_str();

	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);

	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);

	ID = glCreateProgram();
	glAttachShader(ID, vertexShader);
	glAttachShader(ID, fragmentShader);
	glLinkProgram(ID);

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
}

void Shader::_activate() {

	glUseProgram(ID);
}

void Shader::_delete() {

	glDeleteProgram(ID);
}

void Shader::setMat4(const std::string& name, const glm::mat4& mat){
	glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}

void Shader::setVec3(const std::string& name, const glm::vec3& value){
	glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}