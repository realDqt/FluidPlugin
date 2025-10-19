#pragma once
#include "cuda_viewer/cuda_gl_helper.h"

class Shader {
public:
    unsigned int m_shaderId;
    // constructor generates the shader on the fly
    // ------------------------------------------------------------------------
    Shader(const char* vShaderCode, const char* fShaderCode, const char* gShaderCode = nullptr, const char* tcShaderCode = nullptr, const char* teShaderCode = nullptr) {
        // 2. compile shaders
        unsigned int vertex, fragment, geometry, tessControl, tessEvaluation;
        // int success;
        // char infoLog[512];
        // vertex shader
        vertex = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex, 1, &vShaderCode, NULL);
        glCompileShader(vertex);
        checkCompileErrors(vertex, "VERTEX");

        // fragment Shader
        fragment = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment, 1, &fShaderCode, NULL);
        glCompileShader(fragment);
        checkCompileErrors(fragment, "FRAGMENT");

        // geometry Shader
        if (gShaderCode) {
            geometry = glCreateShader(GL_GEOMETRY_SHADER);
            glShaderSource(geometry, 1, &gShaderCode, NULL);
            glCompileShader(geometry);
            checkCompileErrors(geometry, "GEOMETRY");
        }

        if (tcShaderCode && teShaderCode) {
            tessControl = glCreateShader(GL_TESS_CONTROL_SHADER);
            glShaderSource(tessControl, 1, &tcShaderCode, NULL);
            glCompileShader(tessControl);
            checkCompileErrors(tessControl, "TESSELLATION CONTROL");

            tessEvaluation = glCreateShader(GL_TESS_EVALUATION_SHADER);
            glShaderSource(tessEvaluation, 1, &teShaderCode, NULL);
            glCompileShader(tessEvaluation);
            checkCompileErrors(tessEvaluation, "TESSELLATION EVALUATION");
        }


        // shader Program
        m_shaderId = glCreateProgram();
        glAttachShader(m_shaderId, vertex);
        glAttachShader(m_shaderId, fragment);
        if (gShaderCode) glAttachShader(m_shaderId, geometry);
        if (tcShaderCode && teShaderCode) {
            glAttachShader(m_shaderId, tessControl);
            glAttachShader(m_shaderId, tessEvaluation);
        }

        glLinkProgram(m_shaderId);
        checkCompileErrors(m_shaderId, "PROGRAM");

        // delete the shaders as they're linked into our program now and no longer necessary
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        if (gShaderCode) glDeleteShader(geometry);
        if (tcShaderCode && teShaderCode) {
            glDeleteShader(tessControl);
            glDeleteShader(tessEvaluation);
        }
    }

    void use() { glUseProgram(m_shaderId); }

    /**
     * Set a boolean value for a given uniform name in the shader program.
     * 
     * @param name The name of the uniform variable.
     * @param value The boolean value to set.
     */
    void setBool(const std::string& name, bool value) const {
        // Get the location of the uniform variable in the shader program
        glUniform1i(glGetUniformLocation(m_shaderId, name.c_str()), (int)value);
    }

    /**
     * Sets an integer value for a uniform variable in the shader.
     *
     * @param name The name of the uniform variable.
     * @param value The integer value to set.
     */
    void setInt(const std::string& name, int value) const {
        // Get the location of the uniform variable in the shader
        glUniform1i(glGetUniformLocation(m_shaderId, name.c_str()), value);
    }

    /**
     * Set a float value for a given uniform name in the shader program.
     * 
     * @param name The name of the uniform variable.
     * @param value The float value to set.
     */
    void setFloat(const std::string& name, float value) const {
        // Get the location of the uniform variable in the shader program
        glUniform1f(glGetUniformLocation(m_shaderId, name.c_str()), value);
    }

    /**
     * Set a float4 value for a given uniform name in the shader program.
     *
     * @param name The name of the uniform.
     * @param value The float4 value to be set.
     */
    void setFloat4(const std::string& name, float4 value) const {
        // Get the location of the uniform in the shader program
        // std::cout<<"set "<<name<<"="<<value.x<<","<<value.y<<","<<value.z<<","<<value.w<<" at "<<glGetUniformLocation(m_shaderId, name.c_str())<<std::endl;
        glUniform4f(glGetUniformLocation(m_shaderId, name.c_str()), value.x, value.y, value.z, value.w);
    }

    /**
     * Sets the value of a vec3 uniform in the shader program.
     * 
     * @param name The name of the uniform variable.
     * @param value The value to set.
     */
    void setFloat3(const std::string& name, float3 value) const {
        // std::cout<<"set "<<name<<"="<<value.x<<","<<value.y<<","<<value.z<<" at "<<glGetUniformLocation(m_shaderId, name.c_str())<<std::endl;
        glUniform3f(glGetUniformLocation(m_shaderId, name.c_str()), value.x, value.y, value.z);
    }

    // void setVec2(const std::string& name, float value0, float value1) const {
    //     glUniform2f(glGetUniformLocation(m_shaderId, name.c_str()), value0, value1);
    // }
    /**
     * Check if the shader is currently in use.
     *
     * @return true if the shader is in use, false otherwise.
     */
    bool isInUse() const {
        // Get the currently active program
        GLint currentProgram = 0;
        glGetIntegerv(GL_CURRENT_PROGRAM, &currentProgram);
        // Check if the current program is the same as the shader ID
        return (currentProgram == (GLint)m_shaderId);
    }

    // Set up the vertex array for a given attribute
    //
    // Parameters:
    //   name: The name of the attribute
    //   bufferId: The ID of the buffer object
    //   data: The attribute data
    //   step: The number of values per vertex
    //   size: The number of vertices
    //   isDirty: Flag indicating whether the buffer data is dirty
    //
    // Returns:
    //   The location of the attribute in the shader program
    GLint setVertexArray(const std::string& name, GLuint bufferId, float* data, int step, int size, bool isDirty){
        // Print shader ID and attribute name
        std::cout<<"  shaderId "<<m_shaderId<<"name"<<name<<std::endl;
        // Get the location of the attribute in the shader program
        GLint id = glGetAttribLocation(m_shaderId, name.c_str());
        std::cout<<"  get ID "<<id<<" "<<size<<std::endl;
        // If the attribute does not exist in the shader program, return the location
        if (id < 0) return id;
        // If the size is 0, disable the attribute and return the location
        if (size == 0) {
            glDisableVertexAttribArray(id);
            return id;
        }
        // Bind the buffer object
        glBindBuffer(GL_ARRAY_BUFFER, bufferId);
        // If the buffer data is dirty, update the buffer data
        if (isDirty) 
            glBufferData(GL_ARRAY_BUFFER, sizeof(float)*step*size, data, GL_DYNAMIC_DRAW);
        // Specify the attribute format
        glVertexAttribPointer(id, step, GL_FLOAT, GL_FALSE, 0, 0);
        // Enable the attribute
        glEnableVertexAttribArray(id);
        // Return the location of the attribute
        return id;
    }

    /**
     * Set the element array buffer data.
     *
     * @param bufferId The ID of the buffer object.
     * @param data The element array data.
     * @param step The number of values per element.
     * @param size The number of elements.
     * @param isDirty Flag indicating whether the buffer data is dirty.
     *
     * @return The location of the attribute in the shader program.
     */
    GLint setElementArray(GLuint bufferId, unsigned int* data, int step, int size, bool isDirty){
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferId);
        // if (dirty & MeshGL::DIRTY_FACE)
        if (isDirty) 
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*step*size, data, GL_DYNAMIC_DRAW);
        return 0;
    }

private:
    /**
     * Check for compile errors in the shader
     * @param shader The shader to check
     * @param type The type of the shader (VERTEX, FRAGMENT, GEOMETRY, etc.)
     */
    void checkCompileErrors(unsigned int shader, std::string type) {
        int success;
        char infoLog[1024];
        // Check compile status for shaders
        if (type != "PROGRAM") {
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success) {
                // Get the shader info log if compilation fails
                glGetShaderInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
        else {
            // Check link status for program
            glGetProgramiv(shader, GL_LINK_STATUS, &success);
            if (!success) {
                // Get the program info log if linking fails
                glGetProgramInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
        // Exit if there are any compile or link errors
        if (!success) {
            exit(-1);
        }
    }
};