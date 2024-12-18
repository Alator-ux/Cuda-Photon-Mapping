#include "OpenglLayer.h"


std::string read_shader_file(const std::string& filePath) {
    std::ifstream shaderFile(filePath);
    std::stringstream shaderStream;
    shaderStream << shaderFile.rdbuf();
    return shaderStream.str();
}

GLuint compile_shader(const std::string& shaderCode, GLenum shaderType) {
    GLuint shader = glCreateShader(shaderType);
    const char* code = shaderCode.c_str();
    glShaderSource(shader, 1, &code, nullptr);
    glCompileShader(shader);

    // Проверка на ошибки компиляции
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLint logLength;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
        char* log = new char[logLength];
        glGetShaderInfoLog(shader, logLength, &logLength, log);
        std::cerr << "Shader compilation error: " << log << std::endl;
        delete[] log;
    }
    return shader;
}

GLuint create_shader_program_with_sources(const std::string& vertex_shader_code, const std::string& fragment_shader_code) {
    GLuint vertexShader = compile_shader(vertex_shader_code, GL_VERTEX_SHADER);
    GLuint fragmentShader = compile_shader(fragment_shader_code, GL_FRAGMENT_SHADER);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Проверка на ошибки линковки
    GLint success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        GLint logLength;
        glGetProgramiv(shaderProgram, GL_INFO_LOG_LENGTH, &logLength);
        char* log = new char[logLength];
        glGetProgramInfoLog(shaderProgram, logLength, &logLength, log);
        std::cerr << "Program linking error: " << log << std::endl;
        delete[] log;
    }

    // Очистка шейдеров после их линковки
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

GLuint OpenglLayer::create_shader_program(const std::string& vertex_shader_path, const std::string& fragment_shader_path) {
    std::string vertex_source = read_shader_file(vertex_shader_path);
    std::string fragment_source = read_shader_file(fragment_shader_path);

    return create_shader_program_with_sources(vertex_source, fragment_source);
}