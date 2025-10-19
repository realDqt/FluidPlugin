#pragma once
#include "cuda_viewer/shader.h"

extern const char *meshVertexShader;
extern const char *meshFragmentShader;

extern const float g_sphere_vertices[];
extern const float g_sphere_normals[];
extern const unsigned int g_sphere_faces[];

extern const float g_box_vertices[];
extern const float g_box_normals[];
extern const unsigned int g_box_faces[];

extern const float g_cylinder_vertices[];
extern const float g_cylinder_normals[];
extern const unsigned int g_cylinder_faces[];

extern const float g_boxquad_vertices[];
extern const float g_boxquad_normals[];
extern const unsigned int g_boxquad_faces[];

class MeshRenderer {
public:
    enum MeshType{ Triangle=0, Quad=1 };

    MeshRenderer() {
        initShaders();
#if !defined(__APPLE__) && !defined(MACOSX)
        glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
        glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
#endif
        initBuffers();
        m_vertices=nullptr;
        m_normals=nullptr;
        m_elements=nullptr;
        m_vertCnt=0;
        m_elemCnt=0;
    }

    ~MeshRenderer() {
        delete m_shader;
        freeBuffers();
        if(m_vertices) delete[] m_vertices;
        if(m_normals) delete[] m_normals;
        if(m_elements) delete[] m_elements;
        unregisterVbo();
    }

    /**
     * @brief Registers the vertex buffer objects (VBOs) for CUDA processing.
     * 
     * This function registers the specified VBOs for CUDA processing using the
     * `registerGLBufferObject` function. It checks if each VBO is valid before
     * registering it.
     */
    void registerVbo(){
        // registerGLBufferObject(m_vbo, &m_cudaVbo);
        if(m_vertVbo) registerGLBufferObject(m_vertVbo, &m_cudaVertVbo);
        // Register the normal VBO if it exists
        if(m_normVbo) registerGLBufferObject(m_normVbo, &m_cudaNormVbo);
        // Register the element VBO if it exists
        if(m_elemVbo) registerGLBufferObject(m_elemVbo, &m_cudaElemVbo);
        // Register the color VBO if it exists
        if(m_colorVbo) registerGLBufferObject(m_colorVbo, &m_cudaColorVbo);
    }

    // Unregisters the vertex buffer objects (VBOs) used for rendering.
    void unregisterVbo(){
        // unregisterGLBufferObject(m_cudaVbo);
        // if(m_cudaColorVbo) unregisterGLBufferObject(m_cudaColorVbo);
        // Unregister the vertex VBO if it exists.
        if(m_cudaVertVbo) unregisterGLBufferObject(m_cudaVertVbo);
        // Unregister the normal VBO if it exists.
        if(m_cudaNormVbo) unregisterGLBufferObject(m_cudaNormVbo);
        // Unregister the element VBO if it exists.
        if(m_cudaElemVbo) unregisterGLBufferObject(m_cudaElemVbo);
        // Unregister the color VBO if it exists.
        if(m_cudaColorVbo) unregisterGLBufferObject(m_cudaColorVbo);
    }

    /**
     * Initializes the vertex arrays and binds them.
     */
    void beginInitBuffers(){
        glGenVertexArrays(1, &m_vao);
        glBindVertexArray(m_vao);
    }

    /**
     * @brief Unbinds the vertex array object.
     * 
     * This function unbinds the vertex array object by setting it to 0.
     */
    void endInitBuffers(){
        glBindVertexArray(0);
    }

    //// deprecated
    /**
     * @brief Sets the mesh data for rendering.
     * 
     * @param vertices The array of vertex positions.
     * @param normals The array of vertex normals.
     * @param vertCnt The number of vertices.
     * @param elements The array of triangle indices.
     * @param elemCnt The number of elements.
     */
    void setMesh(const float *vertices, const float *normals, int vertCnt, const unsigned int *elements, int elemCnt){
        // Set vertices if provided
        if(vertices && vertCnt>=0){
            if(m_vertices) delete[] m_vertices;
            m_vertices=new float[vertCnt*3];
            memcpy(m_vertices,vertices, vertCnt*3*sizeof(float)); 
            m_vertCnt=vertCnt;
            m_vertDirty=true;
        }else m_vertDirty=false;

        // Set normals if provided
        if(normals && vertCnt>=0){
            if(m_normals) delete[] m_normals;
            m_normals=new float[vertCnt*3];
            memcpy(m_normals,normals, vertCnt*3*sizeof(float)); 
            m_vertDirty=true;
        }else m_vertDirty=false;
        
        // Set elements if provided
        if(elements && elemCnt>=0){
            if(m_elements) delete[] m_elements;
            m_elements=new unsigned int[elemCnt*m_vertPerElem];
            memcpy(m_elements,elements, elemCnt*m_vertPerElem*sizeof(unsigned int)); 
            m_elemCnt=elemCnt;
            m_elemDirty=true;
        }else m_elemDirty=false;

        // Bind vertex array object
        glBindVertexArray(m_vao);

        // Bind and set vertex buffer
        glBindBuffer(GL_ARRAY_BUFFER, m_vertVbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float)*m_vertCnt*3, m_vertices, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	    glEnableVertexAttribArray(0);

        // Bind and set normal buffer
        glBindBuffer(GL_ARRAY_BUFFER, m_normVbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float)*m_vertCnt*3, m_normals, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
	    glEnableVertexAttribArray(1);

        // Bind and set element buffer
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_elemVbo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*m_vertPerElem*m_elemCnt, m_elements, GL_DYNAMIC_DRAW);

        // Unbind vertex array object
        glBindVertexArray(0);
    }

    /**
     * Draws the mesh using OpenGL.
     */
    void draw() {
        // Check if there are any elements to draw
        if(m_elemCnt==0) return;

        // Enable depth testing and depth mask
        glDepthMask(GL_TRUE);
        glEnable(GL_DEPTH_TEST);
        // Use the shader program
        m_shader->use();

        // bindBuffers();//// bind buffer manually
        // Bind the vertex array object
        glBindVertexArray(m_vao);

        // Bind and set up the vertex buffer
        glBindBuffer(GL_ARRAY_BUFFER, m_vertVbo);
        if(m_useCpuData)
            glBufferData(GL_ARRAY_BUFFER, sizeof(float)*m_vertCnt*3, m_vertices, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	    glEnableVertexAttribArray(0);

        // Bind and set up the normal buffer
        glBindBuffer(GL_ARRAY_BUFFER, m_normVbo);
        if(m_useCpuData)
            glBufferData(GL_ARRAY_BUFFER, sizeof(float)*m_vertCnt*3, m_normals, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
	    glEnableVertexAttribArray(1);

        // Bind and set up the element buffer
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_elemVbo);
        if(m_useCpuData)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*m_vertPerElem*m_elemCnt, m_elements, GL_DYNAMIC_DRAW);


        // m_shader->setElementArray(m_elemVbo, m_elements, 3, m_elemCnt, true);
        // Set shader uniforms
        m_shader->setFloat4("fixed_color", m_fixedColor);
        m_shader->setFloat4("position_offset", m_positionOffset);
        m_shader->setFloat3("scale", m_scale);
        // Draw the meshes
        drawMeshes();

        // Unbind the shader program and vertex array object
        glUseProgram(0);
        glBindVertexArray(0);
        // std::cout<<"mesh draw end"<<std::endl;
    }

    /**
     * Set the color to be used by the object.
     *
     * @param color - The color to set.
     */
    void setColor(float4 color){
        m_fixedColor=color;
    }

    /**
     * Set the position offset for the object.
     *
     * @param off - The position offset to set.
     */
    void setPositionOffset(float4 off){
        m_positionOffset=off;
    }

    /**
     * @brief Set the scale of the object.
     * 
     * @param scale - The new scale value.
     */
    void setScale(float3 scale){
        m_scale=scale;
    }

    /**
     * @brief Sets the mesh type.
     * 
     * @param mt The mesh type to set.
     */
    void setMeshType(MeshType mt){
        m_meshType=mt;
        m_vertPerElem=mt+3;
    }

    MeshType getMeshType(){return m_meshType;}

private:
    /**
     * Initialize the shaders.
     */
    void initShaders() {
        // Create a new shader object with the vertex and fragment shader source code
        m_shader = new Shader(meshVertexShader, meshFragmentShader);
    };

    // Initialize the vertex arrays and buffers
    void initBuffers(){
        // Generate a vertex array object
        glGenVertexArrays(1, &m_vao);
        glBindVertexArray(m_vao);
        // Generate vertex, normal, and element buffers
        glGenBuffers(1, &m_vertVbo);
        glGenBuffers(1, &m_normVbo);
        glGenBuffers(1, &m_elemVbo);
        // Unbind the vertex array object
        glBindVertexArray(0);
    }

    /**
     * @brief Frees the OpenGL buffers used for rendering.
     */
    void freeBuffers(){
        // Delete the vertex array object
        glDeleteVertexArrays(1, &m_vao);
        // Delete the vertex buffer object for vertices
        glDeleteBuffers(1, &m_vertVbo);
        // Delete the vertex buffer object for normals
        glDeleteBuffers(1, &m_normVbo);
        // Delete the vertex buffer object for elements
        glDeleteBuffers(1, &m_elemVbo);
    }
    
    // Bind vertex array object
    void bindBuffers(){
        
        glBindVertexArray(m_vao);
        // Set vertex array for position attribute
        m_shader->setVertexArray("position", m_vertVbo, m_vertices, 3, m_vertCnt, true);
        // Set vertex array for normal attribute
        m_shader->setVertexArray("normal", m_normVbo, m_normals, 3, m_vertCnt, true);
        // Set element array
        m_shader->setElementArray(m_elemVbo, m_elements, m_vertPerElem, m_elemCnt, true);
        // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_elemVbo);
        // // F_vbo is data
        // glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned)*F_vbo.size(), F_vbo.data(), GL_DYNAMIC_DRAW);
    }

    /**
     * Draws the meshes with the specified rendering mode.
     * 
     * @param solid Determines whether to render the meshes with solid fill or wireframe lines.
     */
    void drawMeshes(bool solid = true){
        
        // glUniform....
        // Set the rendering mode
        glPolygonMode(GL_FRONT_AND_BACK, solid ? GL_FILL : GL_LINE);

        /* Avoid Z-buffer fighting between filled triangles & wireframe lines */
        if (solid)
        {
            glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(1.0, 1.0);
        }
        // glDrawElements(GL_TRIANGLES, m_elemCnt*3, GL_UNSIGNED_INT, 0);
        // glDrawArrays(GL_TRIANGLES, 0, m_elemCnt*3);
        // Render the meshes
        if(m_meshType==MeshType::Triangle)
        glDrawElements(GL_TRIANGLES, m_elemCnt*3, GL_UNSIGNED_INT, 0);
        else
        glDrawElements(GL_QUADS, m_elemCnt*4, GL_UNSIGNED_INT, 0);

        // Reset rendering settings
        glUseProgram(0);
        glDisable(GL_POLYGON_OFFSET_FILL);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

public:
    Shader* m_shader;

    float* m_vertices;
    float* m_normals;
    unsigned int* m_elements;

    int m_vertCnt;
    int m_elemCnt;

    bool m_vertDirty;
    bool m_elemDirty;
    bool m_useCpuData=false;

    int m_size;

    // float m_radius = 0.05f;
    // float m_fov = 60.0f;

    float4 m_fixedColor=make_float4(1,0,0,0);
    float4 m_positionOffset=make_float4(0,0,0,0);
    float3 m_scale=make_float3(1,1,1);

    int m_winw, m_winh;

    GLuint m_vao=0;
    GLuint m_vertVbo=0;
    GLuint m_normVbo=0;
    GLuint m_elemVbo=0;
    GLuint m_colorVbo=0;

    struct cudaGraphicsResource *m_cudaVertVbo=nullptr;
    struct cudaGraphicsResource *m_cudaNormVbo=nullptr;
    struct cudaGraphicsResource *m_cudaElemVbo=nullptr;
    struct cudaGraphicsResource *m_cudaColorVbo=nullptr;
private:
    MeshType m_meshType=MeshType::Triangle;
    int m_vertPerElem=3;
    // struct cudaGraphicsResource *m_cudaVbo;
    // struct cudaGraphicsResource *m_cudaColorVbo;
};

