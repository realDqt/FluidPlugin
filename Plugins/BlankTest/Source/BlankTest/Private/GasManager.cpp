#include "GasManager.h"
#include "UObject/ConstructorHelpers.h"
#include "object/gas_system.h"
#include "object/grid_gas.h"
#include "object/config.h"
#include "common/timer.h"

using namespace physeng;

uint numParticles = 0;
StopWatchInterface *timer = NULL;
static GasWorld* gasWorld = nullptr;
void TestGasPerformanceDemo(int argc, char** argv)
{
	cudaInit(argc, argv);

	gasWorld = new GasWorld();
	int gasIndex = gasWorld->initGasSystem(make_vec3r(0.0f), 0.000002f, 0.000000f, 4.0f, 5.0f, 0.001f);


	printf("Gas Symbol:%d\n", gasIndex);
	if (gasIndex < 0) {
		exit(0);
	}

	gasWorld->getGas(gasIndex)->addGasSource(make_vec3r(-1.2f, -1.0f, 0.0f), 0.5f, make_vec3r(1.0f, 0.0f, 0.0f), 1.0f);
	gasWorld->getGas(gasIndex)->addBox(make_vec3r(0, -0.7, 0), make_vec3r(0.8, 1.5, 0.8));
	gasWorld->setRenderData(gasIndex, make_vec3r(255 / 255.0f, 255 / 255.0f, 255 / 255.0f), make_float3(0, -1, 0), 0.06f, 100);
	
}

AGasManager::AGasManager()
{
    // 默认关闭 Tick
    PrimaryActorTick.bCanEverTick = true;

    // 创建根组件
    USceneComponent* Root = CreateDefaultSubobject<USceneComponent>(TEXT("RootComponent"));
    RootComponent = Root;

	// 创建用于渲染体积的静态网格体组件
	VolumeMeshComponent = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("VolumeMeshComponent"));
	VolumeMeshComponent->SetupAttachment(RootComponent);
    
	// 加载默认的立方体网格体
	static ConstructorHelpers::FObjectFinder<UStaticMesh> CubeMesh(TEXT("/Engine/BasicShapes/Cube"));
	if (CubeMesh.Succeeded())
	{
		VolumeMeshComponent->SetStaticMesh(CubeMesh.Object);
	}
    
	// Set default performance options
	VolumeMeshComponent->SetMobility(EComponentMobility::Movable);
	VolumeMeshComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
	VolumeMeshComponent->SetCastShadow(false);
	
}

void AGasManager::BeginPlay()
{
    Super::BeginPlay();
    // 准备就绪，等待 UpdateParticlePositions 被调用
	
    TestGasPerformanceDemo(0, nullptr);

	auto gas = gasWorld->getGas(0);
	check(gas)

	// Cache the grid size
	SDKGridSize = gas->m_params.gridSize; //
	if (SDKGridSize.x == 0 || SDKGridSize.y == 0 || SDKGridSize.z == 0)
	{
		UE_LOG(LogTemp, Error, TEXT("SDK Grid Size is zero. Cannot create texture."));
		return;
	}

	// 3. Create the dynamic UVolumeTexture
	
	
}

void AGasManager::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    if (gasWorld && SmokeVolumeTexture)
    {
	    // 1. Run SDK simulation
    	gasWorld->update(0); 
    	auto gas = gasWorld->getGas(0);
    	check(gas)
    	
		//auto& positionDevice = gas->gg.getDensityRef();

		// 2. 让 SDK 将 CPU 密度 (m_hd) 处理为最终的 RGBA 纹理 (m_texture)
		gas->calcRenderData();

    	// 3. 获取 SDK 的 CPU 端 RGBA 纹理数据
    	auto& textureData = gas->getTexture();
    	
    }
    else {
        UE_LOG(LogTemp, Warning, TEXT("fluidWorld == null"));
    }
    //static int curFrame = 0;
    //UE_LOG(LogTemp, Warning, TEXT("Current Frame = %d"), curFrame++);
}

