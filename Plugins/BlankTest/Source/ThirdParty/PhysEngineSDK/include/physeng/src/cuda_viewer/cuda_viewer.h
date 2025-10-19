#pragma once

#include "cuda_viewer/particle_renderer.h"
#include "cuda_viewer/mesh_renderer.h"
#include "cuda_viewer/hair_renderer.h"
#include "cuda_viewer/gas_renderer.h"
#include "common/timer.h"

// Includes
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <functional>

#include <chrono>
#include <thread>


/**
 * @brief This function is responsible for displaying the graphics on the screen.
 */
void display();

/**
 * @brief This function is called when the window is reshaped.
 *
 * @param width The new width of the window.
 * @param height The new height of the window.
 */
void reshape(int width,int height);

/**
 * @brief This function is called when a mouse event occurs.
 *
 * @param button The button that was pressed or released.
 * @param state The state of the button (pressed or released).
 * @param x The x-coordinate of the mouse cursor.
 * @param y The y-coordinate of the mouse cursor.
 */
void mouse(int button, int state, int x, int y);

/**
 * @brief This function is called when the mouse is moved.
 *
 * @param x The new x-coordinate of the mouse cursor.
 * @param y The new y-coordinate of the mouse cursor.
 */
void motion(int x, int y);

/**
 * @brief This function is called when a key is pressed.
 *
 * @param key The ASCII code of the key that was pressed.
 * @param x The x-coordinate of the mouse cursor at the time of the key press.
 * @param y The y-coordinate of the mouse cursor at the time of the key press.
 */
void key(unsigned char key, int x, int y);

/**
 * @brief This function is called when the system is idle.
 */
void idle();

/**
 * @brief This function is called to clean up resources before exiting the program.
 */
void cleanup();

class CudaViewer{
  public:
    /**
     * @brief Default constructor for the CudaViewer class.
     */
    CudaViewer(){}

    /**
     * @brief Destructor for the CudaViewer class.
     *
     * This destructor deletes the dynamically allocated renderers if they exist.
     */
    ~CudaViewer(){
        if(prender) delete prender;
        if(nrender) delete nrender;
        if(mrender) delete mrender;
        if(m2render) delete m2render;
        if(hrender) delete hrender;
        if (grender) delete grender;
    }

    /**
     * @brief Initializes the CudaViewer.
     *
     * @param argc The number of command-line arguments.
     * @param argv An array of command-line argument strings.
     */
    void init(int argc, char **argv);

    // Initialize the OpenGL environment
    void initGL(int *argc, char **argv)
    {
        // Initialize GLUT
        glutInit(argc, argv);
        // Set the display mode with RGB color, depth buffer, double buffering, and multisampling
        glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE | GLUT_MULTISAMPLE);
        // Set the window size
        glutInitWindowSize(width, height);
        // Create the window with the title "CUDA Particles"
        glutCreateWindow("CUDA Particles");
 
        // Check if the required OpenGL version and extensions are supported
        if (!isGLVersionSupported(2,0) ||
            !areGLExtensionsSupported("GL_ARB_multitexture GL_ARB_vertex_buffer_object"))
        {
            fprintf(stderr, "Required OpenGL extensions missing.");
            exit(EXIT_FAILURE);
        }

    #if defined (WIN32)

        // Check if the WGL_EXT_swap_control extension is supported on Windows
        if (wglewIsSupported("WGL_EXT_swap_control"))
        {
            // disable vertical sync
            wglSwapIntervalEXT(0);
        }

    #endif
        // Enable depth testing
        glEnable(GL_DEPTH_TEST);
        // Set the clear color to a dark gray
        glClearColor(0.25, 0.25, 0.25, 1.0);
        // Report any OpenGL errors
        glutReportErrors();
    }

    /**
     * Initializes the CUDA environment.
     *
     * @param argc The number of command-line arguments.
     * @param argv An array of command-line argument strings.
     */
    void initCuda(int argc, char **argv){
        int devID;
        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        devID = findCudaDevice(argc, (const char **)argv);
        if (devID < 0){
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }

// Binds the necessary functions to the glut callbacks
    void bindFunctions() {
        glutDisplayFunc(display); // Set the display function
        glutReshapeFunc(reshape); // Set the reshape function
        glutMouseFunc(mouse); // Set the mouse function
        glutMotionFunc(motion); // Set the motion function
        glutKeyboardFunc(key); // Set the keyboard function
        // glutSpecialFunc(special);
        glutIdleFunc(idle); // Set the idle function
        glutCloseFunc(cleanup); // Set the cleanup function

        // glutMainLoop();
    }

    void run(){
        glutMainLoop();
    }

    /**
     * Creates a new Vertex Buffer Object (VBO) of the specified size.
     *
     * @param size The size of the VBO in bytes.
     * @return The ID of the created VBO.
     */
    static unsigned int createVbo(unsigned int size){
        GLuint vbo;
        // std::cout<<"create vbo size"<<size<<std::endl;
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        return vbo;
    }

    /**
     * Set the boundary of the world.
     * 
     * @param wmin - The minimum point of the world boundary.
     * @param wmax - The maximum point of the world boundary.
     */
    void setWorldBoundary(float3 wmin, float3 wmax){
        world_min_x = wmin.x; //!< Set the minimum x-coordinate of the world
        world_min_y = wmin.y; //!< Set the minimum y-coordinate of the world
        world_min_z = wmin.z; //!< Set the minimum z-coordinate of the world
        world_max_x = wmax.x; //!< Set the maximum x-coordinate of the world
        world_max_y = wmax.y; //!< Set the maximum y-coordinate of the world
        world_max_z = wmax.z; //!< Set the maximum z-coordinate of the world
    }

    // Create a new instance of ParticleRenderer with a parameter of 1 to use normal rendering
    void useNormalRenderer() {
        nrender = new ParticleRenderer(1);
    }

    void useParticleRenderer() {
        // Create a new instance of ParticleRenderer
        prender = new ParticleRenderer();
    }

    // Initialize and use the GasRenderer class
    void useGasRenderer() {
        grender = new GasRenderer();
    }

    /**
     * Enables rendering of the world boundary.
     */
    void useWorldBoundary() {
        renderWorldBoundary = true;
    }

    //// callBacks

    //// camera
  public:
    unsigned int width = 640;
    unsigned int height = 480;

    // view params
    int mouse_x, mouse_y; //!< The current coordinates of the mouse
    int buttonState = 0; //!< The state of the mouse button
    
    float camera_trans[3] = {0, 0, -3}; //!< The translation of the camera
    float camera_rot[3] = { 0, 0, 0 }; //!< The rotation of the camera
    
    float camera_trans_lag[3] = {0, 0, -3}; //!< The lagged translation of the camera
    float camera_rot_lag[3] = {0, 0, 0}; //!< The lagged rotation of the camera
    
    const float inertia = 0.1f; //!< The inertia value for camera movement
    // ParticleRenderer::DisplayMode displayMode = ParticleRenderer::PARTICLE_SPHERES;

    float world_min_x = -10; //!< The minimum x-coordinate of the world
    float world_min_y = 0; //!< The minimum y-coordinate of the world
    float world_min_z = -10; //!< The minimum z-coordinate of the world
    float world_max_x = 10; //!< The maximum x-coordinate of the world
    float world_max_y = 10; //!< The maximum y-coordinate of the world
    float world_max_z = 10; //!< The maximum z-coordinate of the world
    
    bool isPause = false; //!< Flag indicating if the simulation is paused
    bool advanceOneStep = false; //!< Flag indicating if the simulation should advance one step
    bool renderWorldBoundary = false; //!< Flag indicating if the world boundary should be rendered
    // bool demoMode = false;
    // int idleCounter = 0;
    // int demoCounter = 0;
    // const int idleDelay = 2000;
    float modelView[16];
    
    ParticleRenderer* prender; //!< Pointer to a ParticleRenderer object
    ParticleRenderer* nrender; //!< Pointer to another ParticleRenderer object
    MeshRenderer* mrender; //!< Pointer to a MeshRenderer object
    MeshRenderer* m2render; //!< Pointer to another MeshRenderer object
    HairRenderer* hrender; //!< Pointer to a HairRenderer object
    GasRenderer* grender; //!< Pointer to a GasRenderer object

    int mouseButton[3]={0,0,0};
  public:
    // std::function<void()> drawSimulationMenuCallback; //!< Customized menu callback
    std::function<bool(unsigned int, int, int)> keyCallback; //!< Customized key callback
    std::function<bool()> drawCallback; //!< Customized viewer draw callback
    std::function<bool()> updateCallback; //!< Customized viewer update callback
    std::function<bool()> closeCallback; //!< Customized viewer close callback
    
    // Test variables
    float3 startPos; //!< The starting position
    int voxelPerSide; //!< The number of voxels per side
    float length; //!< The length

};