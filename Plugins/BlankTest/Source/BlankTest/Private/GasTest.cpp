#include "GasTest.h"
#include "UObject/ConstructorHelpers.h"
#include "object/gas_system.h"
#include "object/grid_gas.h"
#include "object/config.h"
#include "common/timer.h"
#include "Async/ParallelFor.h"


using namespace physeng;

/*uint numParticles = 0;
uint3 gridSize;
StopWatchInterface *timer = NULL;*/
static GasWorld* gasWorld1 = nullptr;
void TestGasPerformanceDemo1(int argc, char** argv)
{
	printf("Running TestGasPerformanceDemo\n");
	cudaInit(argc, argv);
    printf("Finish cudaInit\n");
	
	gasWorld1 = new GasWorld();
	int gasIndex = gasWorld1->initGasSystem(make_vec3r(0.0f), 0.000002f, 0.000000f, 4.0f, 5.0f, 0.001f);
	
	printf("Gas Symbol:%d\n", gasIndex);
	
	if (gasIndex < 0) {
		exit(0);
	}

	gasWorld1->getGas(gasIndex)->addGasSource(make_vec3r(-1.2f, -1.0f, 0.0f), 0.5f, make_vec3r(1.0f, 0.0f, 0.0f), 1.0f);
	gasWorld1->getGas(gasIndex)->addBox(make_vec3r(0, -0.7, 0), make_vec3r(0.8, 1.5, 0.8));
	gasWorld1->setRenderData(gasIndex, make_vec3r(255 / 255.0f, 255 / 255.0f, 255 / 255.0f), make_float3(0, -1, 0), 0.06f, 100);
	
}

AGasTest::AGasTest()
{
    // 默认关闭 Tick
    PrimaryActorTick.bCanEverTick = true;

    // 创建根组件
    USceneComponent* Root = CreateDefaultSubobject<USceneComponent>(TEXT("RootComponent"));
    RootComponent = Root;

	// [!!] 新增: 创建 InstancedStaticMeshComponent (从 AParticleManager 复制)
	InstancedMeshComponent = CreateDefaultSubobject<UInstancedStaticMeshComponent>(TEXT("InstancedMeshComponent")); //
	InstancedMeshComponent->SetupAttachment(RootComponent); //

	InstancedMeshComponent->SetMobility(EComponentMobility::Movable); //
	InstancedMeshComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision); //
	InstancedMeshComponent->SetCastShadow(false); //

	static ConstructorHelpers::FObjectFinder<UStaticMesh> SphereMesh(TEXT("/Engine/BasicShapes/Sphere")); //
	if (SphereMesh.Succeeded())
	{
		InstancedMeshComponent->SetStaticMesh(SphereMesh.Object); //
	}
	
}

void AGasTest::BeginPlay()
{
    Super::BeginPlay();
    // 准备就绪，等待 UpdateParticlePositions 被调用
	
    TestGasPerformanceDemo1(0, nullptr);

	auto gas = gasWorld1->getGas(0);
	check(gas)

	// Cache the grid size
	//SDKGridSize = gas->m_params.gridSize;
	uint3 SdkSize = gas->m_params.gridSize; 
	SDKGridSize = FIntVector(SdkSize.x, SdkSize.y, SdkSize.z);
	printf("SDKGridSize.x:%d\n", SDKGridSize.X);
	printf("SDKGridSize.y:%d\n", SDKGridSize.Y);
	printf("SDKGridSize.z:%d\n", SDKGridSize.Z);
	if (SDKGridSize.X == 0 || SDKGridSize.Y == 0 || SDKGridSize.Z == 0)
	{
		UE_LOG(LogTemp, Error, TEXT("SDK Grid Size is zero. Cannot create texture."));
		return;
	}

	// 3. Create the dynamic UVolumeTexture

	// [!!] 新增: 缓存坐标转换所需的数据
	SimWorldMin = FVector(gas->m_params.worldMin.x, gas->m_params.worldMin.y, gas->m_params.worldMin.z);
	SimCellLength = gas->m_params.cellLength;

	// [!!] 新增: 应用在蓝图中设置的材质
	if (BaseMaterial)
	{
		InstancedMeshComponent->SetMaterial(0, BaseMaterial); //
	}
	
}

void AGasTest::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    if (gasWorld1)
    {
	    // 1. Run SDK simulation
    	gasWorld1->update(0); 
    	auto gas = gasWorld1->getGas(0);
    	check(gas)
    	
		//auto& positionDevice = gas->gg.getDensityRef();

		// 2. 让 SDK 将 CPU 密度 (m_hd) 处理为最终的 RGBA 纹理 (m_texture)
		gas->calcRenderData();

    	// 3. 获取 SDK 的 CPU 端 RGBA 纹理数据
    	auto& textureData = gas->getTexture();
    	unsigned char* cpuTexture = textureData.m_data;

    	// 4. [!!] 核心修改: 将 3D 纹理(体素) 转换为 粒子位置
    	TArray<FVector> VisibleVoxels;
        
    	// (设置一个阈值，0=全显示, 255=不显示)
    	const unsigned char AlphaThreshold = 10; 

    	for (int z = 0; z < SDKGridSize.Z; z++)
    	{
    		for (int y = 0; y < SDKGridSize.Y; y++)
    		{
    			for (int x = 0; x < SDKGridSize.X; x++)
    			{
    				// 计算 1D 索引
    				int index = (z * SDKGridSize.Y * SDKGridSize.X) + (y * SDKGridSize.X) + x;
    				// 获取 Alpha 通道 (索引 * 4 + 3)
    				unsigned char alpha = cpuTexture[index * 4 + 3];

    				// 如果密度(Alpha)大于阈值
    				if (alpha > AlphaThreshold)
    				{
    					// 5. 将网格索引(x,y,z)转换为世界坐标
    					FVector WorldPos = SimWorldMin + FVector(x, y, z) * SimCellLength;
    					VisibleVoxels.Add(WorldPos);
    				}
    			}
    		}
    	}
        
    	// 6. [!!] 调用 AParticleManager 的渲染函数来绘制这些粒子
    	UpdateParticlePositions(VisibleVoxels);
    	
    }
    else {
        UE_LOG(LogTemp, Warning, TEXT("fluidWorld == null"));
    }
    static int curFrame = 0;
    UE_LOG(LogTemp, Warning, TEXT("Current Frame = %d"), curFrame++);
}

void AGasTest::ClearParticles()
{
	if (InstancedMeshComponent)
	{
		InstancedMeshComponent->ClearInstances(); //
	}
	CurrentInstanceCount = 0; //
}

void AGasTest::UpdateParticlePositions(const TArray<FVector>& NewPositions)
{
	if (!InstancedMeshComponent) //
	{
		return;
	}

	const int32 NewCount = NewPositions.Num(); //

	if (NewCount == 0) //
	{
		if (CurrentInstanceCount > 0) //
		{
			ClearParticles();
		}
		return;
	}

	TransformBuffer.SetNumUninitialized(NewCount); //

	const FQuat RotationAsQuat = FQuat::Identity; //
	// [!!] 注意: 这是一个非常小的缩放，烟雾粒子可能很小
	float scale = 0.01f; //
	const FVector Scale = FVector(scale, scale, scale); //

	ParallelFor(NewCount, [&](int32 i) //
	{
		const FVector& InPos = NewPositions[i]; //
		// [!!] 注意: 您的流体模拟 使用了这个Y/Z轴交换。
		// 您的烟雾模拟很可能也需要它。如果坐标不对，请先尝试移除 -InPos.Z。
		const FVector RotatedPos(InPos.X, -InPos.Z, InPos.Y); //

		TransformBuffer[i].SetComponents(RotationAsQuat, RotatedPos, Scale); //
	});

	UpdateParticleTransforms(TransformBuffer); //
}

void AGasTest::UpdateParticleTransforms(const TArray<FTransform>& NewTransforms)
{
	if (!InstancedMeshComponent) //
	{
		return;
	}

	const int32 NewCount = NewTransforms.Num(); //

	if (NewCount == 0) //
	{
		if (CurrentInstanceCount > 0) //
		{
			ClearParticles();
		}
		return;
	}

	if (NewCount != CurrentInstanceCount) //
	{
		InstancedMeshComponent->ClearInstances(); //
		InstancedMeshComponent->AddInstances(NewTransforms, false); //
	}
	else
	{
		InstancedMeshComponent->BatchUpdateInstancesTransforms(0, NewTransforms, true, true); //
	}

	CurrentInstanceCount = NewCount; //
}

