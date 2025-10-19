#pragma once

#include "cuda_viewer/shader.h"

extern const char* vertexShader;
extern const char* spherePixelShader;
extern const char* vertexShader2;

class Convexcomp
{
private:
    const float3& p0, & up;
public:
    Convexcomp(const float3& p0, const float3& up) : p0(p0), up(up) {}

    /**
     * @brief Determines the ordering of two float3 points based on their relative position to a reference point.
     *
     * @param a The first point to compare.
     * @param b The second point to compare.
     * @return True if point a comes before or is at the same position as point b, false otherwise.
     */
    bool operator()(const float3& a, const float3& b) const
    {
        // Calculate the vectors from the reference point to points a and b
        float3 va = a - p0, vb = b - p0;
        return dot(up, cross(va,vb)) >= 0;
    }
};

class GasRenderer {
public:
    GasRenderer(int i = 0) {
        // cube vertices
        GLfloat cv[][3] = {
            {1.0f, 1.0f, 1.0f}, {-1.0f, 1.0f, 1.0f}, {-1.0f, -1.0f, 1.0f}, {1.0f, -1.0f, 1.0f},
            {1.0f, 1.0f, -1.0f}, {-1.0f, 1.0f, -1.0f}, {-1.0f, -1.0f, -1.0f}, {1.0f, -1.0f, -1.0f}
        };

        GLfloat ce[12][2][3] = {
            {{1.0f, 1.0f, -1.0f}, {0.0f, 0.0f, 1.0f}},
            {{-1.0f, 1.0f, -1.0f}, {0.0f, 0.0f, 1.0f}},
            {{-1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, 1.0f}},
            {{1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, 1.0f}},

            {{1.0f, -1.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
            {{-1.0f, -1.0f, 1.0f}, {0.0f, 1.0f, 0.0f}},
            {{-1.0f, -1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}},
            {{1.0f, -1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}},

            {{-1.0f, 1.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},
            {{-1.0f, -1.0f, 1.0f}, {1.0f, 0.0f, 0.0f}},
            {{-1.0f, -1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}},
            {{-1.0f, 1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}}
        };

        memcpy(_cubeVertices, cv, sizeof(_cubeVertices));
        memcpy(_cubeEdges, ce, sizeof(_cubeEdges));

        initGL();
    }

    ~GasRenderer() {
    }

    /**
     * @brief Fill a 3D texture with data.
     *
     * @param gridSize The size of the grid in the texture.
     * @param textureData Pointer to the texture data.
     */
    void fillTexture(uint3 gridSize, unsigned char* textureData) {
        glActiveTextureARB(GL_TEXTURE0_ARB);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, gridSize.x, gridSize.y, gridSize.z, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData);
    }


    /**
     * This function draws the cube.
     */
    void draw() {
        // Define the array of cube vertices
        GLfloat(*cv)[3] = _cubeVertices;
        int i;
        // Iterate through each vertex of the cube
        for (i = 0; i < 8; i++) {
            // Calculate the transformed coordinates of the vertex
            float x = cv[i][0] + viewDir.x;
            float y = cv[i][1] + viewDir.y;
            float z = cv[i][2] + viewDir.z;
            // Check if the transformed coordinates are within the view range
            if ((x >= -1.0f) && (x <= 1.0f)
                && (y >= -1.0f) && (y <= 1.0f)
                && (z >= -1.0f) && (z <= 1.0f))
            {
                // If the vertex is within the view range, break the loop
                break;
            }
        }
        // Assert that a valid vertex was found within the view range
        assert(i != 8);

        // Define the number of slices for the cube
        float SLICE_NUM = 64.0f;

        // Calculate the distance from the view direction to the current vertex
        float d0 = -dot(viewDir, make_float3(cv[i][0], cv[i][1], cv[i][2]));
        // Calculate the step size for each slice
        float dStep = 2 * d0 / SLICE_NUM;
        // Initialize the slice counter
        int n = 0;
        // Iterate through each slice
        for (float d = -d0; d < d0; d += dStep) {
            // IntersectEdges returns the intersection points of all cube edges with
            // the given plane that lie within the cube
            std::vector<float3> pt = intersectEdges(viewDir.x, viewDir.y, viewDir.z, d);

            // Check if there are enough intersection points to form a polygon
            if (pt.size() > 2) {
                // sort points to get a convex polygon
                std::sort(pt.begin() + 1, pt.end(), Convexcomp(pt[0], viewDir));

                // Enable 3D texture rendering
                glEnable(GL_TEXTURE_3D);
                // Begin drawing a polygon
                glBegin(GL_POLYGON);
                for (i = 0; i < pt.size(); i++) {
                    // Set the color and texture coordinates of the point
                    glColor3f(1.0, 1.0, 1.0);
                    glTexCoord3d((pt[i].x + 1.0) / 2.0, (pt[i].y + 1.0) / 2.0, (pt[i].z + 1.0) / 2.0);//FIXME
                    // Set the vertex coordinates of the point
                    glVertex3f(pt[i].x * gridLength.x / 2.0, pt[i].y * gridLength.y / 2.0, pt[i].z * gridLength.z / 2.0);
                }
                // End drawing the polygon
                glEnd();

            }
            // Increment the slice counter
            n++;
        }
        // Disable 3D texture rendering
        glDisable(GL_TEXTURE_3D);
    }

    /**
     * Set the view direction of the object.
     * 
     * @param viewdir The new view direction to set.
     */
    void setViewDir(float3 viewdir) {
        this->viewDir = normalize(viewdir);
    }

    /**
     * Set the direction of the light source.
     * 
     * @param lightdir The new direction of the light source.
     */
    void setLightDir(float3 lightdir) {
        this->lightDir = normalize(lightdir);
    }

    /**
     * @brief Set the size of the grid and the length of each cell.
     * 
     * @param gridSize The size of the grid in each dimension.
     * @param cellLength The length of each cell.
     */
    void setGridSize(uint3 gridSize, float cellLength) {
        this->gridLength = make_float3(gridSize) * cellLength;
    }


private:
    float3 viewDir;
    float3 lightDir;
    float3 gridLength = make_float3(1.0f);

    unsigned int _hTexture;

    GLfloat _cubeVertices[8][3];
    GLfloat _cubeEdges[12][2][3];

    // Initialize OpenGL settings
    void initGL()
    {
        // Enable 3D texture mapping
        glEnable(GL_TEXTURE_3D);
        // Disable depth testing
        glDisable(GL_DEPTH_TEST);
        // Set front face culling
        glCullFace(GL_FRONT);
        // Enable alpha blending
        glEnable(GL_BLEND);
        // Generate texture IDs
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


        glGenTextures(2, &_hTexture);
        // Activate texture unit 0
        glActiveTextureARB(GL_TEXTURE0_ARB);
        // Bind the 3D texture
        glBindTexture(GL_TEXTURE_3D, _hTexture);
        // Set texture parameter: minification filter
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        // Set texture parameter: wrap mode - s axis
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        // Set texture parameter: wrap mode - t axis
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        // Set texture parameter: wrap mode - r axis
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
    }

    /**
     * Finds the intersection points of a plane and the edges of a cube.
     * 
     * @param A The A component of the plane equation.
     * @param B The B component of the plane equation.
     * @param C The C component of the plane equation.
     * @param D The D component of the plane equation.
     * @return A vector of 3D points representing the intersection points.
     */
    std::vector<float3> intersectEdges(float A, float B, float C, float D)
    {
        // Initialize variables
        float t;
        float3 p;
        std::vector<float3> res;
        GLfloat(*edges)[2][3] = _cubeEdges;

        // Iterate through each edge of the cube
        for (int i = 0; i < 12; i++) {
            t = -(A * edges[i][0][0] + B * edges[i][0][1] + C * edges[i][0][2] + D)
                / (A * edges[i][1][0] + B * edges[i][1][1] + C * edges[i][1][2]);
            if ((t > 0) && (t < 2)) {
                p.x = edges[i][0][0] + edges[i][1][0] * t;
                p.y = edges[i][0][1] + edges[i][1][1] * t;
                p.z = edges[i][0][2] + edges[i][1][2] * t;
                res.push_back(p);
            }
        }
        // Add the intersection point to the result vector
        return res;
    }

public:
};

