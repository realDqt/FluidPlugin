#include "smoke_system.h"
#include "object/particle_fluid.h"
#include "cuda_viewer/cuda_viewer.h"

#include "common/timer.h"

using namespace physeng;
#define GRID_SIZE 64
uint numParticles = 0;
uint3 gridSize;

StopWatchInterface *timer = NULL;

int main(int argc, char **argv) {
    int scene = 0;
    //try {
    //    if(argc==2)
    //        scene = atoi(argv[1]);
    //    else if(argc==3){
    //        scene = atoi(argv[1]);
    //    } else throw 4;
    //}
    //catch (int err) {
    //    LOG_OSTREAM_DEBUG << "error code "<<err<<std::endl;
    //    LOG_OSTREAM_INFO << "Usage: ./FluidCudaDemo [Scene] [PRadius] " << std::endl;
    //    LOG_OSTREAM_INFO << "- Scene ({0,1,2,3}, default 0): 0 for dambreak, 1 for two dambreak, 2 for dambreak with rigid cube, 3 for 70k particle test" << std::endl;
    //    LOG_OSTREAM_INFO << "- PRadius (0.2-0.5, default 0.2): particle radius" << std::endl;
    //    // LOG_OSTREAM_INFO << "- PhaseType ({1,4}, default 1): 1 for liquid, 2 for oil, 3 for sand" << std::endl;
    //    LOG_OSTREAM_INFO << "Key Control:" << std::endl;
    //    LOG_OSTREAM_INFO << "- space: start/pause" << std::endl;
    //    LOG_OSTREAM_INFO << "- a: step" << std::endl;
    //    LOG_OSTREAM_INFO << "- mouse L: rotate camera" << std::endl;
    //    LOG_OSTREAM_INFO << "- mouse R: zoom in/out" << std::endl;
    //    return 0;
    //}

    cudaInit(argc, argv);
    uint3 gridSize; gridSize.x = gridSize.z = GRID_SIZE;
    gridSize.y = GRID_SIZE * 4;
    Real cellLength = 0.05;
    SmokeSystem *psystem = nullptr;

    //switch(scene){
    //    //// smoke
    //    case 0:
    psystem = new SmokeSystem(gridSize, cellLength, make_vec3r(gridSize.x, 0, gridSize.z) * -cellLength, make_vec3r(gridSize.x * 2, gridSize.y * 2, gridSize.z * 2) * cellLength);
    psystem->m_params.kvorticity = 5;
    psystem->m_params.gravity = make_vec3r(0, -9.8, 0);
    psystem->updateParams();
    //        break;
    //    default:
    //        LOG_OSTREAM_ERROR<<"invalid scene"<<std::endl;
    //        break;
    //}


    LOG_OSTREAM_DEBUG<<"finish add fluid"<<std::endl;

    std::vector<int3>showPos;
    Real* den = new Real[psystem->gf.getDensityRef().size()];
    copyArray<Real, MemType::CPU, MemType::GPU>(&den, &psystem->gf.getDensityRef().m_data, 0, psystem->gf.getDensityRef().size());
    for (int i = 0; i < psystem->gf.getDensityRef().size(); i++) {
        if (den[i] > 0.2)
            showPos.push_back(_d_getGridIdx(i, psystem->m_params.gridHashMultiplier));
    }
    if (showPos.size() == 0)
        showPos.push_back(make_int3(-1000, -1000, -1000));
    vec3r* showDen = new vec3r[showPos.size()];
    for (int i = 0; i < showPos.size(); i++) {
        showDen[i] = make_vec3r(showPos[i]) * psystem->m_params.cellLength;
    }

    CudaViewer viewer;
    viewer.init(argc, argv);
    viewer.bindFunctions();
    viewer.camera_trans[2] = -60;

    viewer.prender->m_pos = (float*)showDen;
    viewer.prender->m_size = showPos.size();
    viewer.prender->m_radius = psystem->getCellLength() / 2;
    
    viewer.prender->m_vbo = viewer.createVbo(viewer.prender->m_size * 3 * sizeof(Real));
    viewer.prender->m_colorVbo = viewer.createVbo(viewer.prender->m_size * 3 * sizeof(Real));
    viewer.prender->registerVbo();

    viewer.setWorldBoundary(psystem->m_params.worldMin, psystem->m_params.worldMax);

    sdkCreateTimer(&timer);
    viewer.keyCallback = [&](unsigned int key, int, int) {
        LOG_OSTREAM_DEBUG << "Key down " << (char)key << std::endl;
        return false;
    };

    delete[] den;
    delete[] showDen;
    int frame=0;

    viewer.updateCallback = [&]() {
        std::vector<int3>showPos;
        Real* den = new Real[psystem->gf.getDensityRef().size()];
        copyArray<Real, MemType::CPU, MemType::GPU>(&den, &psystem->gf.getDensityRef().m_data, 0, psystem->gf.getDensityRef().size());
        for (int i = 0; i < psystem->gf.getDensityRef().size(); i++) {
            if (den[i] > 0.2)
                showPos.push_back(_d_getGridIdx(i, psystem->m_params.gridHashMultiplier));
        }
        if (showPos.size() == 0)
            showPos.push_back(make_int3(-1000, -1000, -1000));
        vec3r* showDen = new vec3r[showPos.size()];
        for (int i = 0; i < showPos.size(); i++) {
            showDen[i] = make_vec3r(showPos[i]) * psystem->m_params.cellLength;
        }

        viewer.prender->m_size = showPos.size();
        glDeleteBuffers(1, &viewer.prender->m_vbo);
        glDeleteBuffers(1, &viewer.prender->m_colorVbo);
        viewer.prender->m_vbo = viewer.createVbo(viewer.prender->m_size * 3 * sizeof(Real));
        viewer.prender->m_colorVbo = viewer.createVbo(viewer.prender->m_size * 3 * sizeof(Real));
        viewer.prender->unregisterVbo();
        viewer.prender->registerVbo();

        Real dt=0.016;

        sdkStartTimer(&timer);
        {
            // PHY_PROFILE("grid system update");
            psystem->update(dt);
            frame++;
        }
        {
            // PHY_PROFILE("copy buffer");
            vec3r* cudaPtr = (vec3r*)mapGLBufferObject(&viewer.prender->m_cudaVbo);
            copyArray<vec3r, MemType::GPU, MemType::CPU>(&(cudaPtr), &showDen, showPos.size());
            unmapGLBufferObject(viewer.prender->m_cudaVbo);
        }
        sdkStopTimer(&timer);

        {
            // PHY_PROFILE("copy buffer");
            vec3r* cudaPtr = (vec3r*)mapGLBufferObject(&viewer.prender->m_cudaColorVbo);
            copyArray<vec3r, MemType::GPU, MemType::GPU>(&(cudaPtr), &(psystem->getColorRef()).m_data, showPos.size());
            unmapGLBufferObject(viewer.prender->m_cudaColorVbo);
        }

        if(frame%100==0){
            float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
            LOG_OSTREAM_INFO<<"frame "<<frame<<", fps="<<ifps<<", time="<<sdkGetTimerValue(&timer) / 1000.f<<std::endl;
            BENCHMARK_REPORT();
        }
        delete[] den;
        delete[] showDen;
        return true;
    };

    viewer.closeCallback = [&](){
        sdkDeleteTimer(&timer);
        delete psystem;
        return true;
    };
    viewer.isPause=true;
    viewer.run();
}