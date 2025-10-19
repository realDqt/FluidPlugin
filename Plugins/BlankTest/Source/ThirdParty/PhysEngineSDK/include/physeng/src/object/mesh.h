#pragma once

#include <vector>
#include "common/allocator.h"
#include "math/math.h"

struct Mesh
{
    /**
     * @brief Add a mesh.
     * 
     * @param m The mesh to add.
     */
    void AddMesh(const Mesh& m);

    /**
     * @brief Normalize the mesh.
     */
    void Normalize();

    /**
     * @brief Returns the number of vertices in the object.
     *
     * @return The number of vertices.
     */
    uint32_t GetNumVertices() const { return uint32_t(m_positions.size()); }

    /**
     * @brief Returns the number of faces in the mesh.
     * 
     * @return The number of faces.
     */
    uint32_t GetNumFaces() const { return uint32_t(m_indices.size()) / 3; }

    /**
     * @brief Duplicate a vertex.
     */
	void DuplicateVertex(uint32_t i);

    std::vector<vec3r> m_positions;
    std::vector<vec3r> m_normals;
    std::vector<vec2r> m_texcoords[2];
    std::vector<vec3r> m_colours;

    std::vector<int> m_indices;    
};

/**
 * @brief Import a mesh from an OBJ file.
 * 
 * @param path The path to the OBJ file.
 * @return A pointer to the imported mesh.
 */
Mesh* ImportMeshFromObj(const char* path);

/**
 * @brief Import a mesh from a PLY file.
 *
 * @param path The path to the PLY file.
 * @return A pointer to the imported mesh.
 */
Mesh* ImportMeshFromPly(const char* path);

/**
 * @brief Imports a mesh from a binary file.
 * 
 * @param path The path to the binary file.
 * @return A pointer to the imported mesh.
 */
Mesh* ImportMeshFromBin(const char* path);

/**
 * @brief Switches on filename
 *
 * @param path The file path of the mesh to import.
 * @return A pointer to the imported mesh.
 */
Mesh* ImportMesh(const char* path);

/**
 * @brief Returns the file type of a given path.
 *
 * @param path The path of the file.
 * @return The file type as a string.
 */
std::string GetFileType(const char* path);