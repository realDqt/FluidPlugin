#include "GasManager.h"
#include "UObject/ConstructorHelpers.h"
#include "object/gas_system.h"
#include "object/grid_gas.h"
#include "object/config.h"
#include "common/timer.h"

#ifdef UpdateResource
#undef UpdateResource
#endif
#include "Engine/VolumeTexture.h"

using namespace physeng;

uint numParticles = 0;
StopWatchInterface *timer = NULL;
static GasWorld* gasWorld = nullptr;
void TestGasPerformanceDemo(int argc, char** argv)
{
	printf("Running TestGasPerformanceDemo\n");
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
	// PF_R8G8B8A8 对应 SDK 的 unsigned char RGBA 缓冲区
	// 3a. 创建 UObject 实例
	SmokeVolumeTexture = NewObject<UVolumeTexture>(
			this,       // Outer
			NAME_None,  // Name
			RF_Transient // Flags (瞬态的，不会被保存)
		);

	// 3b. 创建并配置 PlatformData
	SmokeVolumeTexture->PlatformData = new FTexturePlatformData();
	SmokeVolumeTexture->PlatformData->SizeX = SDKGridSize.x;
	SmokeVolumeTexture->PlatformData->SizeY = SDKGridSize.y;
	SmokeVolumeTexture->PlatformData->SetNumSlices(SDKGridSize.z); // 设置深度
	SmokeVolumeTexture->PlatformData->PixelFormat = PF_R8G8B8A8;    // 设置像素格式

	// 3c. 创建唯一的 MipMap 级别
	FTexture2DMipMap* Mip = new FTexture2DMipMap();
	SmokeVolumeTexture->PlatformData->Mips.Add(Mip);
	Mip->SizeX = SDKGridSize.x;
	Mip->SizeY = SDKGridSize.y;
        
	// 3d. 为 MipMap 分配内存
	const int32 TextureDataSize = SDKGridSize.x * SDKGridSize.y * SDKGridSize.z * 4; // 4 bytes (RGBA)
	Mip->BulkData.Lock(LOCK_READ_WRITE);
	void* DestData = Mip->BulkData.Realloc(TextureDataSize);
        
	// (可选，但推荐) 将新分配的内存清零，避免初始闪烁
	FMemory::Memzero(DestData, TextureDataSize); 
        
	Mip->BulkData.Unlock();

	// 3e. 初始化纹理资源
	SmokeVolumeTexture->UpdateResource();

	// 4. 创建和配置动态材质实例 (MID)
	if (BaseVolumeMaterial)
	{
		// 从蓝图中设置的基础材质创建动态实例
		DynamicVolumeMaterial = UMaterialInstanceDynamic::Create(BaseVolumeMaterial, this);
        
		// 将我们的动态纹理设置为材质中的参数(注意：材质参数名必须是 'SmokeTexture')
		DynamicVolumeMaterial->SetTextureParameterValue(FName("SmokeTexture"), SmokeVolumeTexture);
        
		// 将动态材质应用到立方体网格上
		VolumeMeshComponent->SetMaterial(0, DynamicVolumeMaterial);
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("AGasManager: 'BaseVolumetricMaterial' is not set in the Blueprint! Cannot render smoke."));
	}
    
	// 5. 缩放和定位渲染立方体以匹配模拟边界
	FVector worldMin = FVector(gas->m_params.worldMin.x, gas->m_params.worldMin.y, gas->m_params.worldMin.z); //
	FVector worldMax = FVector(gas->m_params.worldMax.x, gas->m_params.worldMax.y, gas->m_params.worldMax.z); //
    
	// The default cube is 100x100x100 units. We scale it to match the world size.
	FVector WorldSize = worldMax - worldMin;
	FVector WorldCenter = worldMin + (WorldSize / 2.0f);
    
	// UE 的默认立方体是 100x100x100。缩放因子 = 世界大小 / 100
	VolumeMeshComponent->SetWorldScale3D(WorldSize / 100.0f);
	VolumeMeshComponent->SetRelativeLocation(WorldCenter);

	
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
    	
    	// 4. 锁定 UE 纹理资源以进行写入
    	FTexturePlatformData* PlatformData = SmokeVolumeTexture->PlatformData;
    	check(PlatformData);

    	if (PlatformData->Mips.Num() == 0)
    	{
    		UE_LOG(LogTemp, Error, TEXT("SmokeVolumeTexture Mips.Num() is 0."));
    		return;
    	}
    	
    	FTexture2DMipMap& Mip = PlatformData->Mips[0];
    	void* DestTextureData = Mip.BulkData.Lock(LOCK_READ_WRITE);

    	// 5. 将 C++ 数组数据复制到 UE 纹理
    	// 我们假设 VecArray 有一个 .m_data 成员指向原始 unsigned char*
    	// (基于你在 gas_system.cpp 中看到的 m_hd.m_data)
    	const void* SourceTextureData = textureData.m_data; 
    	const int32 TextureDataSize = SDKGridSize.x * SDKGridSize.y * SDKGridSize.z * 4; // 4 bytes (RGBA)

    	FMemory::Memcpy(DestTextureData, SourceTextureData, TextureDataSize);

    	// 6. 解锁并更新纹理，使其在GPU上生效
    	Mip.BulkData.Unlock();
    	SmokeVolumeTexture->UpdateResource();
    }
    else {
        UE_LOG(LogTemp, Warning, TEXT("fluidWorld == null"));
    }
    //static int curFrame = 0;
    //UE_LOG(LogTemp, Warning, TEXT("Current Frame = %d"), curFrame++);
}

