#include "GasManager.h"
#include "UObject/ConstructorHelpers.h"
#include "object/gas_system.h"
#include "object/grid_gas.h"
#include "object/config.h"
#include "common/timer.h"
#include "cuda_runtime.h"
#include "cuda_d3d11_interop.h" 
#include "D3D11RHI.h"

#ifdef UpdateResource
#undef UpdateResource
#endif
#include "Engine/VolumeTexture.h"

using namespace physeng;

uint numParticles = 0;
//uint3 gridSize;
StopWatchInterface *timer = NULL;
static GasWorld* gasWorld = nullptr;
void TestGasPerformanceDemo(int argc, char** argv)
{
	printf("Running TestGasPerformanceDemo\n");
	cudaInit(argc, argv);
    printf("Finish cudaInit\n");
	
	gasWorld = new GasWorld();
	int gasIndex = gasWorld->initGasSystem(make_vec3r(0.0f), 0.000002f, 0.000000f, 4.0f, 5.0f, 0.001f);
	
	printf("Gas Symbol:%d\n", gasIndex);
	
	if (gasIndex < 0) {
		exit(0);
	}

	gasWorld->getGas(gasIndex)->addGasSource(make_vec3r(-1.2f, -1.0f, 0.0f), 0.5f, make_vec3r(1.0f, 0.0f, 0.0f), 1.0f);
	gasWorld->getGas(gasIndex)->addBox(make_vec3r(0, -0.7, 0), make_vec3r(0.8, 1.5, 0.8));
	gasWorld->setRenderData(gasIndex, make_vec3r(255 / 255.0f, 255 / 255.0f, 255 / 255.0f), make_float3(0, -1, 0), 0.06f, 100);
	printf("TestGasPerformanceDemo\n");
	
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

	// 初始化成员变量
	/*gasWorld = nullptr;
	gasIndex = -1;
	bShouldCreateTexture = false;
	bTextureCreated = false;*/
}

void AGasManager::BeginPlay()
{
    Super::BeginPlay();
    // 准备就绪，等待 UpdateParticlePositions 被调用
	
    TestGasPerformanceDemo(0, nullptr);
	/*gasWorld = new GasWorld();
	if (!gasWorld)
	{
		printf("gasWorld is nullptr");
		UE_LOG(LogTemp, Error, TEXT("Failed to create GasWorld instance"));
		return;
	}
    
	// 确检查 gasIndex，而不是使用 exit(0)
	gasIndex = gasWorld->initGasSystem(make_vec3r(0.0f), 0.000002f, 0.000000f, 4.0f, 5.0f, 0.001f);
	if (gasIndex < 0)
	{
		UE_LOG(LogTemp, Error, TEXT("GasWorld SDK initGasSystem() 失败! 返回索引: %d"), gasIndex);
		return; // 安全地退出 BeginPlay，Tick 将不会运行
	}

	//UE_LOG(LogTemp, Log, TEXT("Gas Symbol:%d"), gasIndex);
	printf("Gas Symbol:%d\n", gasIndex);

	//SDK 的其他设置
	gasWorld->getGas(gasIndex)->addGasSource(make_vec3r(-1.2f, -1.0f, 0.0f), 0.5f, make_vec3r(1.0f, 0.0f, 0.0f), 1.0f);
	gasWorld->getGas(gasIndex)->addBox(make_vec3r(0, -0.7, 0), make_vec3r(0.8, 1.5, 0.8));
	gasWorld->setRenderData(gasIndex, make_vec3r(255 / 255.0f, 255 / 255.0f, 255 / 255.0f), make_float3(0, -1, 0), 0.06f, 100);*/

	auto gas = gasWorld->getGas(0);
	check(gas)

	// Cache the grid size
	SDKGridSize = gas->m_params.gridSize; //
	printf("SDKGridSize.x:%d\n", SDKGridSize.x);
	printf("SDKGridSize.y:%d\n", SDKGridSize.y);
	printf("SDKGridSize.z:%d\n", SDKGridSize.z);
	
	if (SDKGridSize.x <= 0 || SDKGridSize.y <= 0 || SDKGridSize.z <= 0)
	{
		UE_LOG(LogTemp, Error, TEXT("SDK Grid Size is zero. Cannot create texture."));
		return;
	}
    
	UE_LOG(LogTemp, Log, TEXT("AGasManager::BeginPlay - Initialization complete, waiting for first tick to create texture"));

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
	Mip->SizeZ = SDKGridSize.z;
        
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
		//DynamicVolumeMaterial->SetScalarParameterValue(FName("Opacity"), 1.0f); // 确保不透明度为 1
		//DynamicVolumeMaterial->SetVectorParameterValue(FName("Color"), FLinearColor(1, 1, 1)); // 确保颜色为白色
        
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
	VolumeMeshComponent->SetWorldScale3D(WorldSize);
	VolumeMeshComponent->SetRelativeLocation(WorldCenter);

	UE_LOG(LogTemp, Warning, TEXT("Grid %d×%d×%d  TextureDataSize=%d"), 
	SDKGridSize.x,SDKGridSize.y,SDKGridSize.z, TextureDataSize);

	UE_LOG(LogTemp,Warning,TEXT("WorldSize %s"), *WorldSize.ToString());
	UE_LOG(LogTemp,Warning,TEXT("WorldCenter %s"), *WorldSize.ToString());
	
	
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

    	// =====================================================
    	// [DEBUG START] 验证数据是否全为 0
    	// =====================================================
    	const unsigned char* ByteData = static_cast<const unsigned char*>(SourceTextureData);
    	bool bHasData = false;
    	int32 NonZeroCount = 0;
    	int32 coloredCount = 0;
    	unsigned char MaxAlpha = 0;

    	// 为了性能，我们可以只检查每第 4 个字节 (Alpha 通道)，或者检查前几千个字节
    	// 这里为了绝对确认，我们检查整个数组 (注意：在 Release 模式下这非常快，但在 Debug 模式下可能会卡顿)
    	for (int32 i = 3; i < TextureDataSize; i += 4) // 从索引 3 开始，步长 4，只检查 Alpha
    	{
    		if (ByteData[i] > 0)
    		{
    			bHasData = true;
    			NonZeroCount++;
    			if (ByteData[i] > MaxAlpha) MaxAlpha = ByteData[i];
    		}
    	}
    	for (int32 i = 0; i < TextureDataSize; i += 4) // 改为从 0 开始，步长 4，检查 R 通道
    	{
    		// 检查 R, G, B 任意一个是否有值
    		if (ByteData[i] > 0 || ByteData[i+1] > 0 || ByteData[i+2] > 0) 
    		{
    			// 这是一个有颜色的像素！
    			coloredCount++;
    		}
    	}

    	// 限制日志输出频率，每 60 帧打印一次，避免刷屏
    	static int32 LogCounter = 0;
    	if (LogCounter++ % 60 == 0) 
    	{
    		if (!bHasData)
    		{
    			// 如果打印这个，说明 CUDA 模拟静默失败了，或者没有产生任何密度
    			UE_LOG(LogTemp, Error, TEXT("[数据验证] 纹理数据全为 0 (Alpha通道)！模拟未运行或无烟雾生成。"));
    		}
    		else
    		{
    			// 如果打印这个，说明模拟正常！问题一定出在材质或纹理设置上
    			UE_LOG(LogTemp, Warning, TEXT("[数据验证] 发现数据！非 0 体素数量: %d, 最大 Alpha 值: %d, 有颜色的像素块数量: %d"), NonZeroCount, MaxAlpha, coloredCount);
    		}
    	}
    	// =====================================================
    	// [DEBUG END]
    	// =====================================================
    	
    	FMemory::Memcpy(DestTextureData, SourceTextureData, TextureDataSize);
    
    	//FMemory::Memset(DestTextureData, 128, TextureDataSize);  // RGBA=128 → 中灰
    	

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

