#include "gas_world.h"

Logger logger("gasLog", 1024);

GasWorld::GasWorld()
{
    dt = 0.1f;
}

GasWorld::~GasWorld()
{
    for (auto gas : gasArray) {
        if (gas) delete gas;
    }
}

int GasWorld::initGasSystem(vec3r origin, Real vorticity, Real diffusion, Real buoyancy, Real vcEpsilon, Real decreaseDensity)
{
    uint3 gridSize = make_uint3(GAS_GRID_SIZE, GAS_GRID_SIZE, GAS_GRID_SIZE);
    Real cellLength = 0.05f;
    vec3r worldMin = -make_vec3r(gridSize.x, gridSize.y, gridSize.z) * cellLength * 0.5;
    vec3r worldMax = make_vec3r(gridSize.x, gridSize.y, gridSize.z) * cellLength * 0.5;

    if (origin.x < worldMin.x || origin.x > worldMax.x ||
        origin.y < worldMin.y || origin.y > worldMax.y ||
        origin.z < worldMin.z || origin.z > worldMax.z) {
        printf("???��?��??????����?????????????��???????\n");
        logger.Log(LogType::Error, "???????����?????????????�^????��???"+std::to_string(origin.x)+"," + std::to_string(origin.y) + "," + std::to_string(origin.z));
        return -1;
    }

    if (vorticity < 0 || diffusion < 0 || vcEpsilon < 0 || decreaseDensity < 0) {
        printf("�������󣬲����������壬����ʧ��\n");
        logger.Log(LogType::Error, "gas's property is abnormal");
        return -1;
    }
    LOG_OSTREAM_DEBUG << "add gas at " << "(" << origin.x << "," << origin.y << "," << origin.z <<  ")" << std::endl;
    LOG_OSTREAM_DEBUG << "vorticity: " << vorticity << " diffusion: " << diffusion << " buoyancy: " << buoyancy << " vcEpsilon: " << vcEpsilon << " decreaseDensity: " << decreaseDensity << std::endl;
    GasSystem* gasSystem = new GasSystem(gridSize, cellLength, worldMin, worldMax);
    this->origin = origin;

    gasSystem->m_params.kvorticity = vorticity;
    gasSystem->m_params.kdiffusion = diffusion;
    gasSystem->m_params.kbuoyancy = buoyancy;
    gasSystem->m_params.vc_eps = vcEpsilon;
    gasSystem->setDecreaseDensity(decreaseDensity);

    gasSystem->updateParams();


    gasSystem->setColor(make_vec3r(238 / 255.0f, 221 / 255.0f, 153 / 255.0f));
    gasSystem->setLightDir(make_float3(0, 0, -1));
    gasSystem->generateRayTemplate();


    gasArray.push_back(gasSystem);

    int id = gasArray.size() - 1;
    logger.Log(LogType::Info, "gas id:" + std::to_string(id));
    return id;
}

int GasWorld::initNuclearSystem(vec3r origin, Real vorticity) {
    uint3 gridSize = make_uint3(NUCLEAR_GRID_SIZE, NUCLEAR_GRID_SIZE, NUCLEAR_GRID_SIZE);
    Real cellLength = 0.05f;
    NuclearSystem* nuclearSystem =  new NuclearSystem(gridSize, cellLength, make_vec3r(0, 0, 0) * -cellLength, make_vec3r(gridSize.x, gridSize.y, gridSize.z) * cellLength);

    nuclearSystem->m_params.kvorticity = vorticity;
    //gravity
    nuclearSystem->m_params.gravity = make_vec3r(0.0, -9.8, 0);
    nuclearSystem->updateParams();
    

    nuclearArray.push_back(nuclearSystem);
    useGas = false;

    int id = nuclearArray.size() - 1;
    logger.Log(LogType::Info, "gas id:" + std::to_string(id));
    return id;
}

int GasWorld::initGasSystem(const char* config_file)
{
    Config config(config_file);

    Real vorticity, diffusion, buoyancy, vcEpsilon, decreaseDensity;
    vec3r origin;
    GET_PARAM(config, vorticity);
    GET_PARAM(config, diffusion);
    GET_PARAM(config, buoyancy);
    GET_PARAM(config, vcEpsilon);
    GET_PARAM(config, decreaseDensity);
    GET_PARAM(config, origin);
    return initGasSystem(origin, vorticity, diffusion, buoyancy, vcEpsilon, decreaseDensity);
}

GasSystem* GasWorld::getGas(int index) {
    return gasArray[index];
}

void GasWorld::setRenderData(int index, vec3r color, vec3r lightDir, Real decay, unsigned char ambient) {
    gasArray[index]->setColor(color);
    gasArray[index]->setLightDir(lightDir);
    gasArray[index]->setDecay(decay);
    gasArray[index]->setAmbient(ambient);
    gasArray[index]->generateRayTemplate();
}

Real GasWorld::getAverageDensity(int index){
    return gasArray[index]->getAverageDensity();
}
void GasWorld::initViewer(int index, CudaViewer& viewer) {
    if (useGas) {
        GasSystem* gsystem = gasArray[index];
        viewer.setWorldBoundary(gsystem->m_params.worldMin, gsystem->m_params.worldMax);
        //viewer.useWorldBoundary();

        viewer.useGasRenderer();
        viewer.grender->setLightDir(gsystem->getLightDir());
        viewer.grender->setGridSize(gsystem->m_params.gridSize, gsystem->m_params.cellLength);

        viewer.grender->fillTexture(gsystem->m_params.gridSize, gsystem->getTexture().m_data);
    }
    else {
        NuclearSystem* nsystem = nuclearArray[index];

        viewer.useNormalRenderer();
        viewer.useParticleRenderer();

        std::vector<int3>showPos;
        Real* den = nsystem->getDensity().m_data;
        for (int i = 0; i < nsystem->gf.getDensityRef().size(); i++) {
            if (den[i] > 0.2)//threshold = 0.2
                showPos.push_back(_d_getGridIdx(i, nsystem->m_params.gridHashMultiplier));
        }
        if (showPos.size() == 0)
            showPos.push_back(make_int3(-1000, -1000, -1000));
        vec3r* showDen = new vec3r[showPos.size()];
        for (int i = 0; i < showPos.size(); i++) {
            showDen[i] = make_vec3r(showPos[i]) * nsystem->m_params.cellLength;
        }

        viewer.nrender->use = true;
        viewer.prender->m_pos = (float*)showDen;
        viewer.prender->den = den;
        viewer.prender->m_size = showPos.size();
        viewer.prender->m_radius = nsystem->getCellLength() / 2;

        viewer.prender->m_vbo = viewer.createVbo(viewer.prender->m_size * 3 * sizeof(Real));
        viewer.prender->m_colorVbo = viewer.createVbo(viewer.prender->m_size * 3 * sizeof(Real));
        viewer.prender->registerVbo();

        viewer.setWorldBoundary(nsystem->m_params.worldMin, nsystem->m_params.worldMax);
    }

}

void GasWorld::updateViewer(int index, CudaViewer& viewer) {
    if (useGas) {
        GasSystem* gsystem = gasArray[index];
        gsystem->calcRenderData();
        viewer.grender->fillTexture(gsystem->m_params.gridSize, gsystem->getTexture().m_data);
    }
    else {
        NuclearSystem* nsystem = nuclearArray[index];
        std::vector<int3>showPos;

        //this is set to record the color type
        std::vector<int> colorRecord;


        Real* den = nsystem->getDensity().m_data;
        for (int i = 0; i < nsystem->gf.getDensityRef().size(); i++) {
            if (den[i] > 1.0) {
                //red color, set record = 1
                showPos.push_back(_d_getGridIdx(i, nsystem->m_params.gridHashMultiplier));
                colorRecord.push_back(1);
            }
            else if (den[i] > 0.5) {
                //blue color, set record = 2
                showPos.push_back(_d_getGridIdx(i, nsystem->m_params.gridHashMultiplier));
                colorRecord.push_back(2);
            }
            else if (den[i] > 0.1) {
                //green color, set record = 3
                showPos.push_back(_d_getGridIdx(i, nsystem->m_params.gridHashMultiplier));
                colorRecord.push_back(3);
            }
        }
        if (showPos.size() == 0)
            showPos.push_back(make_int3(-1000, -1000, -1000));
        int recordSize = colorRecord.size();
        vec3r* renderColor;

        //if(recordSize != 0){
        renderColor = new vec3r[recordSize];
        //}
        for (int i = 0; i < recordSize; ++i) {
            switch (colorRecord[i])
            {
            case 1:
                /* code */
                //red color
                renderColor[i] = make_vec3r(0.9, 0.0, 0.0);
                break;
            case 2:
                //blue color  
                renderColor[i] = make_vec3r(0.0, 0.0, 0.9);
                break;
            case 3:
                //green color
                renderColor[i] = make_vec3r(0.0, 0.9, 0.0);
                break;
            default:
                printf("impossible!\n");
            }
        }

        vec3r* showDen = new vec3r[showPos.size()];



        for (int i = 0; i < showPos.size(); i++) {
            showDen[i] = make_vec3r(showPos[i]) * nsystem->m_params.cellLength;
        }


        viewer.prender->m_size = showPos.size();


        glDeleteBuffers(1, &viewer.prender->m_vbo);
        glDeleteBuffers(1, &viewer.prender->m_colorVbo);


        viewer.prender->m_vbo = viewer.createVbo(viewer.prender->m_size * 3 * sizeof(Real));
        viewer.prender->m_colorVbo = viewer.createVbo(viewer.prender->m_size * 3 * sizeof(Real));

        viewer.prender->unregisterVbo();
        viewer.prender->registerVbo();

        {
            //this is passed to GPU, show the particle that is needed to be rendered
            // PHY_PROFILE("copy buffer");
            vec3r* cudaPtr = (vec3r*)mapGLBufferObject(&viewer.prender->m_cudaVbo);
            // vec3r* cudaPtr = (vec3r*)mapGLBufferObject(&viewer.prender->m_cudaVbo_r);
            copyArray<vec3r, MemType::GPU, MemType::CPU>(&(cudaPtr), &showDen, showPos.size());
            // vec3r* cudaPtr = (vec3r*)mapGLBufferObject(&viewer.prender->m_cudaVbo_r);
            // copyArray<vec3r, MemType::GPU, MemType::CPU>(&(cudaPtr), &showDen_r, showPos_r.size());
            unmapGLBufferObject(viewer.prender->m_cudaVbo);
        }


        {
            // PHY_PROFILE("copy buffer");
            // this is passed to GPU to record the color
            vec3r* cudaPtr = (vec3r*)mapGLBufferObject(&viewer.prender->m_cudaColorVbo);
            // vec3r* cudaPtr = (vec3r*)mapGLBufferObject(&viewer.prender->m_cudaColorVbo_r);
            //do not use color here beacuse originla color is too rough

            copyArray<vec3r, MemType::GPU, MemType::CPU>(&(cudaPtr), &renderColor, showPos.size());
            // copyArray<vec3r, MemType::GPU, MemType::GPU>(&(cudaPtr), &(psystem->getColorRef()).m_data, showPos.size());
            // copyArray<vec3r, MemType::GPU, MemType::GPU>(&(cudaPtr), &(psystem->getColorRef()).m_data, showPos_r.size());
            unmapGLBufferObject(viewer.prender->m_cudaColorVbo);
        }
        delete[] renderColor;
        delete[] showDen;

    }
}

void GasWorld::setIfDissipate(int index, bool flag) {
    gasArray[index]->setIfGenerate(flag);
}

void GasWorld::update(int index) {
    if (useGas) {
        gasArray[index]->update(dt);
    }
    else {
        nuclearArray[index]->update(dt);
    }
}

void GasWorld::setFireSmoke(int index) {
    GasSystem* gsystem = gasArray[index];
    gsystem->addGasSource(origin + make_vec3r(0.0f, -(gsystem->m_params.gridSize.y / 2.0f * gsystem->m_params.cellLength), 0.0f), 0.4f, make_vec3r(0.0f, 1.0f, 0.0f), 1.0f);

    gsystem->addGasSource(origin + make_vec3r(-(7 * gsystem->m_params.cellLength), -(gsystem->m_params.gridSize.y / 2.0f * gsystem->m_params.cellLength), (8 * gsystem->m_params.cellLength)), 0.15f, make_vec3r(-0.2f, 1.0f, 0.1f), 0.5f);
    gsystem->addGasSource(origin + make_vec3r((6 * gsystem->m_params.cellLength), -(gsystem->m_params.gridSize.y / 2.0f * gsystem->m_params.cellLength), -(4 * gsystem->m_params.cellLength)), 0.1f, make_vec3r(0.1f, 1.0f, -0.2f), 0.3f);
    gsystem->addGasSource(origin + make_vec3r((3 * gsystem->m_params.cellLength), -(gsystem->m_params.gridSize.y / 2.0f * gsystem->m_params.cellLength), (5 * gsystem->m_params.cellLength)), 0.2f, make_vec3r(0.2f, 1.0f, 0.2f), 0.4f);
    //gsystem->addGasSource(make_vec3r(1.6f, -0.8f, 0.0f), 0.2f, make_vec3r(-2.0f, 0.0f, 0.0f), 1.0f);
    gsystem->m_params.kdiffusion = 0.000001f;
    gsystem->m_params.kvorticity = 0.000001f;
    gsystem->updateParams();

    setRenderData(index, make_vec3r(100 / 255.0f, 100 / 255.0f, 100 / 255.0f), make_float3(0, -1, 0), 0.06f, 100);
}

void GasWorld::setExplodeSmoke(int index) {
    GasSystem* gsystem = gasArray[index];
    gsystem->addGasSource(origin + make_vec3r(0.0f, -(10 * gsystem->m_params.cellLength), 0.0f), 0.4f, make_vec3r(0.0f, 0.4f, 0.0f), 1.0f);

    gsystem->addGasSource(origin + make_vec3r(-(7 * gsystem->m_params.cellLength), -(6 * gsystem->m_params.cellLength), (8 * gsystem->m_params.cellLength)), 0.15f, make_vec3r(-0.08f, 0.1f, 0.04f), 0.5f);
    gsystem->addGasSource(origin + make_vec3r((6 * gsystem->m_params.cellLength), -(8 * gsystem->m_params.cellLength), -(4 * gsystem->m_params.cellLength)), 0.1f, make_vec3r(0.04f, 0.03f, -0.08f), 0.3f);
    gsystem->addGasSource(origin + make_vec3r((3 * gsystem->m_params.cellLength), -(13 * gsystem->m_params.cellLength), (5 * gsystem->m_params.cellLength)), 0.2f, make_vec3r(0.08f, -0.2f, 0.08f), 0.4f);
    gsystem->addGasSource(origin + make_vec3r(-(5 * gsystem->m_params.cellLength), -(10 * gsystem->m_params.cellLength), -(7 * gsystem->m_params.cellLength)), 0.2f, make_vec3r(-0.08f, -0.2f, -0.08f), 0.6f);
    //gsystem->addGasSource(make_vec3r(1.6f, -0.8f, 0.0f), 0.2f, make_vec3r(-2.0f, 0.0f, 0.0f), 1.0f);
    gsystem->m_params.kdiffusion = 0.00001f;
    gsystem->m_params.kvorticity = 0.000001f;
    gsystem->m_params.kbuoyancy = 0.5f;
    gsystem->updateParams();

    setRenderData(index, make_vec3r(100 / 255.0f, 100 / 255.0f, 100 / 255.0f), make_float3(0, -1, 0), 0.06f, 100);
}

void GasWorld::setExhaust(int index) {
    GasSystem* gsystem = gasArray[index];

    gsystem->addGasSource(origin + make_vec3r(-(gsystem->m_params.gridSize.x / 2 * gsystem->m_params.cellLength), -(gsystem->m_params.gridSize.x / 4 * gsystem->m_params.cellLength), 0.0f), 0.4f, make_vec3r(0.4f, -0.2f, 0.0f), 1.0f);
    //gsystem->addGasSource(make_vec3r(1.6f, -0.8f, 0.0f), 0.2f, make_vec3r(-2.0f, 0.0f, 0.0f), 1.0f);
    gsystem->m_params.kdiffusion = 0.000001f;
    gsystem->m_params.kvorticity = 0.000001f;
    gsystem->m_params.kbuoyancy = 0.5f;
    gsystem->updateParams();

    gsystem->setAlpha(0.015f);
    setRenderData(index, make_vec3r(255 / 255.0f, 255 / 255.0f, 255 / 255.0f), make_float3(0, -1, 0), 0.06f, 100);
}

void GasWorld::setDust(int index) {
    GasSystem* gsystem = gasArray[index];

    gsystem->addGasSource(origin + make_vec3r(-(gsystem->m_params.gridSize.x / 2 * gsystem->m_params.cellLength), -(gsystem->m_params.gridSize.x / 2 * gsystem->m_params.cellLength), 0.0f), 0.4f, make_vec3r(0.5f, 0.4f, 0.0f), 1.0f);
    //gsystem->addGasSource(make_vec3r(1.6f, -0.8f, 0.0f), 0.2f, make_vec3r(-2.0f, 0.0f, 0.0f), 1.0f);
    gsystem->m_params.kdiffusion = 0.00001f;
    gsystem->m_params.kvorticity = 0.000001f;
    gsystem->m_params.kbuoyancy = -0.4f;
    gsystem->updateParams();

    gsystem->setMaxLife(20.0f);
    gsystem->setAlpha(0.02f);
    setRenderData(index, make_vec3r(255 / 255.0f, 248 / 255.0f, 220 / 255.0f), make_float3(0, -1, 0), 0.06f, 100);
}

void GasWorld::setBiochemistry(int index) {
    GasSystem* gsystem = gasArray[index];

    gsystem->addGasSource(origin + make_vec3r(-1.2, -1.2f, 0), 0.4f, make_vec3r(0.1f, 0.0f, 0.0f), 1.0f);
    gsystem->m_params.kdiffusion = 0.0000001f;
    gsystem->m_params.kvorticity = 0.0000000f;
    gsystem->m_params.kbuoyancy = 0.0f;
    gsystem->m_params.vc_eps = 2.0f;
    gsystem->setDecreaseDensity(0.0000f);
    gsystem->updateParams();
    gsystem->setAlpha(0.1f);
    setRenderData(index, make_vec3r(100 / 255.0f, 200 / 255.0f, 0 / 255.0f), make_float3(0, -1, 0), 0.06f, 100);
}

void GasWorld::setWind(Real strength, vec3r direction) {
    if (strength < 0.0f) {
        logger.Log(LogType::Error, "gas's property is abnormal");
        exit(-1);
    }
    if (useGas) {
        if (length(direction) > 1e-7) {
            for (auto gas : gasArray) {
                gas->setWindDirection(normalize(direction));
                gas->setWindStrength(strength);
            }
        }
    }
    else {
        for (auto nuclear : nuclearArray) {
            nuclear->setNormWind(strength);
        }
    }

}