#include "fluid_system.h"
#include "object/particle_fluid.h"
#include "cuda_viewer/cuda_viewer.h"

#include "common/timer.h"

using namespace physeng;
#define GRID_SIZE 64
uint numParticles = 0;
uint3 gridSize;

StopWatchInterface *timer = NULL;

//FluidSystem* psystem = 0;
int main(int argc, char **argv) {
    Real radius = 0.2f;
    int scene = 0;
    try {
        if(argc==2)
            scene = atoi(argv[1]);
        else if(argc==3){
            scene = atoi(argv[1]);
            radius = atof(argv[2]);
        } else throw 4;

        if (radius<0.2 || radius>0.5) throw 1;
        //if (phase!=PhaseType::Liquid && phase!=PhaseType::Oil && phase!=PhaseType::Sand && phase!=PhaseType::Rigid) throw 3;
    }
    catch (int err) {
        LOG_OSTREAM_DEBUG << "error code "<<err<<std::endl;
        LOG_OSTREAM_INFO << "Usage: ./FluidCudaDemo [Scene] [PRadius] " << std::endl;
        LOG_OSTREAM_INFO << "- Scene ({0,1,2,3}, default 0): 0 for dambreak, 1 for two dambreak, 2 for dambreak with rigid cube, 3 for 70k particle test" << std::endl;
        LOG_OSTREAM_INFO << "- PRadius (0.2-0.5, default 0.2): particle radius" << std::endl;
        // LOG_OSTREAM_INFO << "- PhaseType ({1,4}, default 1): 1 for liquid, 2 for oil, 3 for sand" << std::endl;
        LOG_OSTREAM_INFO << "Key Control:" << std::endl;
        LOG_OSTREAM_INFO << "- space: start/pause" << std::endl;
        LOG_OSTREAM_INFO << "- a: step" << std::endl;
        LOG_OSTREAM_INFO << "- mouse L: rotate camera" << std::endl;
        LOG_OSTREAM_INFO << "- mouse R: zoom in/out" << std::endl;
        return 0;
    }

    cudaInit(argc, argv);
    uint3 gridSize; gridSize.x = gridSize.y = gridSize.z = GRID_SIZE;
    FluidSystem *psystem = nullptr;

    switch(scene){
        //// liquid
        case 10:
            // 70000 particle dam break, performance
            psystem=new FluidSystem(131072, radius, gridSize, make_vec3r(-72,0,-72)*radius, make_vec3r(72,128,72)*radius, true);
            psystem->addDam(make_vec3r(-20,52,-20)*radius, make_vec3r(35-0.001,50-0.001,40-0.001)*radius*1.8, radius*1.8, PhaseType::Liquid);
            psystem->m_params.kvorticity=0.f;
            psystem->m_params.kviscosity=0.05f;
            psystem->updateParams();
            break;
        case 11:
            // dambreak, compared with Flex
            psystem=new FluidSystem(131072, radius, gridSize, make_vec3r(0,0,-5.58-5.58), make_vec3r(33.32,32,5.58), true);
            psystem->addDam(make_vec3r(5.58,11.16,0), make_vec3r(11.15,22.32,11.15), radius*1.8, PhaseType::Liquid);
            // psystem->addTerrain();
            psystem->m_params.kviscosity=0.05f;
            psystem->m_params.kvorticity=0.0f;
            psystem->setSubSteps(2);
            psystem->setSolverIterations(3);
            psystem->updateParams();
            break;
        case 12:
            // two dambreak
            psystem=new FluidSystem(131072, radius, gridSize, make_vec3r(-16,0,-16), make_vec3r(16,40,16), true);
            psystem->addDam(make_vec3r(-11,6,-11), make_vec3r(10.1,10.1,10.1), radius*1.8, PhaseType::Liquid);
            psystem->addDam(make_vec3r(11,6,11), make_vec3r(10.1,10.1,10.1), radius*1.8, PhaseType::Liquid);
            break;
        case 13:
            // dambreak with collider
            psystem=new FluidSystem(131072, radius, gridSize, make_vec3r(0,0,-8.2), make_vec3r(32.0,32,8.2));
            psystem->addDam(make_vec3r(4.,8.,0.), make_vec3r(7.98,16,15.98), radius*1.8, PhaseType::Liquid);
            psystem->m_params.kviscosity=0.05f;
            psystem->m_params.kvorticity=3.0f;
            psystem->m_params.useColumnObstacle=true;
            psystem->updateColumn(0, 2, make_vec3r(16, 0, 0));
            psystem->updateColumn(1, 2, make_vec3r(16, 0, 5));
            psystem->updateColumn(2, 2, make_vec3r(16, 0, -5));
            psystem->updateParams();
            break;
        case 14:
            // dambreak with large vorticity
            psystem=new FluidSystem(131072, radius, gridSize, make_vec3r(0,0,-4.2), make_vec3r(24.4,32,4.2));
            psystem->addDam(make_vec3r(4.,8.,0.), make_vec3r(7.98,16,7.98), radius*1.8, PhaseType::Liquid);
            psystem->m_params.kviscosity=0.05f;
            psystem->m_params.kvorticity=6.0f;
            psystem->updateParams();
            break;
        case 15:
            // two phase flow
            psystem=new FluidSystem(131072, radius, gridSize, make_vec3r(-25,0,-25)*radius, make_vec3r(25,72,25)*radius, false);
            psystem->addDam(make_vec3r(12,32,12)*radius, make_vec3r(24,48,24)*radius, radius*1.8, PhaseType::Liquid);
            psystem->addDam(make_vec3r(-12,32,-12)*radius, make_vec3r(24,48,24)*radius, radius*1.8, PhaseType::Oil);
            psystem->setSolverIterations(5);
            psystem->m_params.kviscosity=0.1f;
            psystem->updateParams();
            break;
        case 16:
            // surface tension
            psystem=new FluidSystem(131072, radius, gridSize, make_vec3r(-64,-64,-64)*radius, make_vec3r(64,64,64)*radius, true);
            psystem->addDam(make_vec3r(-0,0,-0)*radius, make_vec3r(33,65,33)*radius, radius*1.8, PhaseType::Liquid);
            psystem->m_params.gravity=make_vec3r(0,0,0);
            psystem->m_params.kviscosity=0.05f;
            psystem->updateParams();
            break;
        case 17:
            // liquid with rotate stick
            psystem=new FluidSystem(131072, radius, gridSize, make_vec3r(-8.2,0,-8.2), make_vec3r(8.2,32,8.2));
            psystem->addDam(make_vec3r(5.,8.,0.), make_vec3r(3.98,16,15.98), radius*1.8, PhaseType::Liquid);
            psystem->addDam(make_vec3r(-5.,8.,0.), make_vec3r(3.98,16,15.98), radius*1.8, PhaseType::Liquid);
            psystem->m_params.kviscosity=0.05f;
            psystem->m_params.kvorticity=4.0f;
            psystem->m_params.useColumnObstacle=true;
            psystem->updateColumn(0, 2, make_vec3r(0, 0, -5));
            psystem->updateColumn(1, 2, make_vec3r(0, 0, 5));
            psystem->updateParams();
            break;

        //// sand
        case 30:
            // 75000 sand particle dam break, performance
            psystem=new FluidSystem(131072, radius, gridSize, make_vec3r(-96,0,-96)*radius, make_vec3r(96,128,96)*radius);
            psystem->addDam(make_vec3r(-20,52,0)*radius, make_vec3r(15-0.001,50-0.001,50-0.001)*radius*2.0, radius*2.0, PhaseType::Sand);
            psystem->addDam(make_vec3r(20,52,0)*radius, make_vec3r(15-0.001,50-0.001,50-0.001)*radius*2.0, radius*2.0, PhaseType::Sand);
            psystem->m_params.staticFriction=50.f;
            psystem->m_params.dynamicFriction=50.f;
            // psystem->m_params.sleepVelocity=9.81f*0.002f;
            psystem->m_params.sleepVelocity=radius*0.2f;
            psystem->m_params.sleepVelocity2=psystem->m_params.sleepVelocity*psystem->m_params.sleepVelocity;
            psystem->setSolverIterations(10);
            psystem->updateParams();
            break;
        case 31:
            // sand dambreak
            psystem=new FluidSystem(131072, radius, gridSize, make_vec3r(0,0,-5), make_vec3r(24.4,32,5));
            psystem->addDam(make_vec3r(4.,8.,0.), make_vec3r(7.98,16,7.98), radius*2.0, PhaseType::Sand);
            psystem->setSolverIterations(3);
            psystem->updateParams();
            break;
        case 32:
            // sand with rotate stick
            psystem=new FluidSystem(131072, radius, gridSize, make_vec3r(-8.2,0,-8.2), make_vec3r(8.2,32,8.2));
            psystem->addDam(make_vec3r(5.,8.,0.), make_vec3r(3.98,16,15.98), radius*2.0, PhaseType::Sand);
            psystem->addDam(make_vec3r(-5.,8.,0.), make_vec3r(3.98,16,15.98), radius*2.0, PhaseType::Sand);
            psystem->m_params.useColumnObstacle=true;
            psystem->updateColumn(0, 2, make_vec3r(0, 0, -5));
            psystem->updateColumn(1, 2, make_vec3r(0, 0, 5));
            psystem->updateParams();
            break;
        case 33:
            // dambreak with a cube
            psystem=new FluidSystem(131072, radius, gridSize, make_vec3r(-15,0,-15), make_vec3r(15,40,15), true);
            psystem->addDam(make_vec3r(-8,7,-8), make_vec3r(12.1,12.1,12.1), radius*2.0, PhaseType::Liquid);

            psystem->addCubeFromMesh(make_vec3r(5,10,0), make_vec3r(4.0), radius*2.0, PhaseType::Rigid);
            psystem->addModel("C:/MyCode/data/bunny.ply", make_vec3r(-5, 10, 6), make_vec3r(4.0), radius * 2.0, PhaseType::Rigid);
            psystem->addTerrain();
            break;
        case 34:
            // dambreak with moving boundary
            psystem=new FluidSystem(131072, radius, gridSize, make_vec3r(-15,0,-15), make_vec3r(15,40,15));
            psystem->addDam(make_vec3r(-6,6,-6), make_vec3r(10.1,10.1,10.1), radius*2.0, PhaseType::Sand);
            psystem->addDam(make_vec3r(6,6,6), make_vec3r(10.1,10.1,10.1), radius*2.0, PhaseType::Sand);
            break;
        default:
            LOG_OSTREAM_ERROR<<"invalid scene"<<std::endl;
            break;
    }


    LOG_OSTREAM_DEBUG<<"fin add fluid"<<std::endl;

    CudaViewer viewer;
    viewer.init(argc, argv);
    viewer.bindFunctions();
    viewer.camera_trans[2] = -60;
    viewer.prender->m_pos = (float*)(psystem->pf.getPositionRef()).m_data;
    viewer.prender->m_size = (psystem->pf.getPositionRef()).size();
    viewer.prender->m_radius = psystem->getParticleRadius();
    
    viewer.prender->m_vbo = viewer.createVbo(viewer.prender->m_size * 3 * sizeof(Real));
    viewer.prender->m_colorVbo = viewer.createVbo(viewer.prender->m_size * 3 * sizeof(Real));
    viewer.prender->registerVbo();
    // viewer.mrender->setMesh(g_cylinder_vertices,g_cylinder_normals,192,g_cylinder_faces,64);
    sdkCreateTimer(&timer);
    viewer.keyCallback = [&](unsigned int key, int, int) {
        LOG_OSTREAM_DEBUG << "Key down " << (char)key << std::endl;
        return false;
    };

    int frame=0;

    viewer.updateCallback = [&]() {
        Real dt=0.016;

        {
            if(scene==17||scene==32){
                Real time=frame*dt;
                psystem->updateColumn(0, 2, 5.0f*make_vec3r(-sinr(time), 0, -cosr(time)));
                psystem->updateColumn(1, 2, 5.0f*make_vec3r(sinr(time), 0, cosr(time)));
                psystem->updateParams();
            }

            if(scene==11){
                Real time=frame*dt;
                
                Real x=12*sinr(time);
                psystem->m_params.worldMin.x=-x;
                psystem->updateParams();
            }

            if(scene==34){
                Real time=frame*dt;
                
                Real x=12+6*cosr(time/2);
                psystem->m_params.worldMin.x=-x;
                psystem->m_params.worldMin.z=-x;
                psystem->m_params.worldMax.x=x;
                psystem->m_params.worldMax.z=x;
                psystem->setSubSteps(2);
                psystem->setSolverIterations(2);
                psystem->updateParams();
            }
        }
        viewer.setWorldBoundary(psystem->m_params.worldMin, psystem->m_params.worldMax);

        sdkStartTimer(&timer);
        {
            // PHY_PROFILE("particle system update");
            psystem->update(dt);
            frame++;
        }
        {
            // PHY_PROFILE("copy buffer");
            vec3r* cudaPtr = (vec3r*)mapGLBufferObject(&viewer.prender->m_cudaVbo);
            copyArray<vec3r, MemType::GPU, MemType::GPU>(&(cudaPtr), &(psystem->pf.getPositionRef()).m_data, (psystem->pf.getPositionRef()).size());
            unmapGLBufferObject(viewer.prender->m_cudaVbo);
        }
        sdkStopTimer(&timer);

        {
            // PHY_PROFILE("copy buffer");
            vec3r* cudaPtr = (vec3r*)mapGLBufferObject(&viewer.prender->m_cudaColorVbo);
            copyArray<vec3r, MemType::GPU, MemType::GPU>(&(cudaPtr), &(psystem->getColorRef()).m_data, (psystem->pf.getPositionRef()).size());
            unmapGLBufferObject(viewer.prender->m_cudaColorVbo);
        }

        if(frame%100==0){
            float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
            LOG_OSTREAM_INFO<<"frame "<<frame<<", fps="<<ifps<<", time="<<sdkGetTimerValue(&timer) / 1000.f<<std::endl;
            if(psystem->m_params.useFoam)
            LOG_OSTREAM_INFO<<"foamParticle "<<psystem->getFoamParticleCount()<<std::endl;
            BENCHMARK_REPORT();
        }
        return true;
    };

    viewer.closeCallback = [&](){
        sdkDeleteTimer(&timer);
        delete psystem;
        return true;
    };
    viewer.isPause=true;
    // registerGLBufferObject(viewer.prender->m_vbo, &viewer.prender->m_cudaVbo);
    viewer.run();
}