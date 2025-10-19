#pragma once

#include "cuda_viewer/shader.h"

extern const char *vertexShader;
extern const char *spherePixelShader;
extern const char *vertexShader2;

class ParticleRenderer {
public:
    public:
    /**
     * @brief Constructor for ParticleRenderer.
     *
     * @param i An integer parameter.
     *        If i is 0, initialize shaders using initShaders().
     *        If i is 1, initialize shaders using initShaders2().
     */
    ParticleRenderer(int i = 0) {
        use = false;
        if(i == 0)
            initShaders();
        else if (i == 1){
            initShaders2();
        }
#if !defined(__APPLE__) && !defined(MACOSX)
        glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
        glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
#endif
    }

    /**
     * @brief Destructor for ParticleRenderer.
     *
     * Deletes the shader object, sets m_pos to 0, and calls unregisterVbo().
     */
    ~ParticleRenderer() {
        delete m_shader;
        m_pos = 0;
        unregisterVbo();
    }

    /**
     * @brief Register VBO and color VBO with CUDA
     */
    void registerVbo(){
        // Register VBO with CUDA
        registerGLBufferObject(m_vbo, &m_cudaVbo);
        // Check if color VBO exists
        if(m_colorVbo) registerGLBufferObject(m_colorVbo, &m_cudaColorVbo);
    }

    /**
     * @brief Unregisters the VBO and color VBO.
     * 
     * This function unregisters the VBO and color VBO if they are registered.
     * 
     */
    void unregisterVbo(){
        // Unregister the VBO
        unregisterGLBufferObject(m_cudaVbo);
        // Check if the color VBO is registered and unregister it
        if(m_cudaColorVbo) unregisterGLBufferObject(m_cudaColorVbo);
    }

    /**
     * Enable necessary OpenGL features for drawing.
     * Set shader uniforms for point size, scale, and radius.
     * Draw particles.
     * Disable OpenGL features.
     */
    void draw() {
        // Enable point sprites and replace texture coordinates
        glEnable(GL_POINT_SPRITE_ARB);
        glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
        // Enable vertex program point size and depth mask
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
        glDepthMask(GL_TRUE);
        // Enable depth testing
        glEnable(GL_DEPTH_TEST);

        // Use the shader program
        m_shader->use();
        // Set the point scale based on the window height and field of view
        m_shader->setFloat("pointScale", float(m_winh) / tanf(m_fov * 0.5f * (float)3.1415926 / 180.0f));
        // m_shader->setVec2("screenSize", m_winw,m_winh);
        m_shader->setFloat("pointRadius", m_radius);

        // Set the point radius
        glColor3f(1, 1, 1);
        // Draw the particles
        drawParticles();

        glUseProgram(0);
        // Disable point sprites
        glDisable(GL_POINT_SPRITE_ARB);
    }


private:
    /**
     * @brief Initialize the shaders for drawing particles.
     * 
     * This function creates a new Shader object using the vertexShader and spherePixelShader.
     * It assigns the newly created Shader object to the m_shader member variable.
     */
    void initShaders() {
        m_shader = new Shader(vertexShader, spherePixelShader);
    };

    // Initialize the shaders used for rendering
    void initShaders2(){
        // Create a new shader object with the provided vertex and pixel shaders
        m_shader = new Shader(vertexShader2,spherePixelShader);
    };

    /**
     * Draw particles using OpenGL.
     */
    void drawParticles() {
        // std::cout<<"vbo="<<m_vbo<<",cvbo="<<m_colorVbo<<",size="<<m_size<<" "<<std::endl;
        // Check if the vertex buffer object (VBO) is initialized
        if (!m_vbo) {
            // Draw particles using immediate mode
            glBegin(GL_POINTS);
            {
                int k = 0;
                for (int i = 0; i < m_size; ++i) {
                    glVertex3fv(&m_pos[k]);
                    k += 4;
                }
            }
            glEnd();
        }
        else {
            // Bind the vertex buffer object (VBO)
            glBindBuffer(GL_ARRAY_BUFFER, m_vbo);////TODO: 4->3?
            // glVertexPointer(4, GL_FLOAT, 0, 0);
            // Set the vertex pointer
            glVertexPointer(3, GL_FLOAT, 0, 0);
            glEnableClientState(GL_VERTEX_ARRAY);

            if (m_colorVbo) {
                // Bind the color buffer object (CBO)
                glBindBuffer(GL_ARRAY_BUFFER, m_colorVbo);
                // glColorPointer(4, GL_FLOAT, 0, 0);
                // Set the color pointer
                glColorPointer(3, GL_FLOAT, 0, 0);
                glEnableClientState(GL_COLOR_ARRAY);
            }
            // Draw particles using vertex arrays
            glDrawArrays(GL_POINTS, 0, m_size);

            // Unbind the buffers and disable the client state
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glDisableClientState(GL_VERTEX_ARRAY);
            glDisableClientState(GL_COLOR_ARRAY);
        }
    }


public:
    Shader* m_shader;  ///< Pointer to the shader object used for rendering.
    
    float* m_pos;  ///< Pointer to the array storing particle positions.
    // add den to control color
    float* den;  ///< Pointer to the array storing density values.
    
    int m_size = 0;  ///< The size of the particle arrays.
    
    float m_radius = 0.05f;  ///< The radius of the particles.
    float m_fov = 60.0f;  ///< The field of view for rendering.
    
    int m_winw, m_winh;  ///< The width and height of the rendering window.
    
    GLuint m_vbo = 0;  ///< The vertex buffer object (VBO) used for storing particle positions.
    GLuint m_colorVbo = 0;  ///< The color buffer object (CBO) used for storing particle colors.
    
    bool use;  ///< Flag indicating whether the particle renderer is in use or not.
    
    struct cudaGraphicsResource* m_cudaVbo = nullptr;  ///< CUDA graphics resource for the particle VBO.
    struct cudaGraphicsResource* m_cudaColorVbo = nullptr;  ///< CUDA graphics resource for the color VBO.


};

