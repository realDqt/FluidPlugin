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
StopWatchInterface *timer = NULL;
//uint3 gridSize;


AGasManager::AGasManager()
{
	FGuid guid = GetActorGuid();
	UE_LOG(LogTemp, Warning, TEXT("AGasManager"));
	//UE_LOG(LogTemp, Warning, TEXT("AGasManager"));
	
    // 默认关闭 Tick
    PrimaryActorTick.bCanEverTick = true;

    // 创建根组件
    USceneComponent* Root = CreateDefaultSubobject<USceneComponent>(TEXT("RootComponent"));
    RootComponent = Root;
}

void AGasManager::BeginPlay()
{
	UE_LOG(LogTemp, Warning, TEXT("AGasManager::BeginPlay"));
    Super::BeginPlay();

	//init gasWorld
	gasWorld = new GasWorld();
	if (!gasWorld)
	{
		printf("gasWorld is nullptr");
		UE_LOG(LogTemp, Warning, TEXT("Failed to create GasWorld instance"));
		return;
	}

	//gasWorldParams
	//initGasSystem(make_vec3r(0.0f), 0.000002f, 0.000000f, 4.0f, 5.0f, 0.001f);
	
}

AUEGasSyetem* AGasManager::CreateAndSpawnGasSystem(FVector Location)
{
	UE_LOG(LogTemp, Warning, TEXT("SpawnGasSystem - Location: %s"), *Location.ToString());
	
	if (!gasWorld && !GasSystemClass)
	{
		UE_LOG(LogTemp, Warning, TEXT("gasWorld or gasSystemClass is nullpter"));
		return nullptr;
	}
	
	//GasWorldParams
	//TODO::
	//initGasSystem可编辑
	int newIndex = gasWorld->initGasSystem(make_vec3r(0.0f), 0.000002f, 0.000000f, 4.0f, 5.0f, 0.001f);

	if (newIndex < 0) return nullptr;
	ActiveGasSystemIDs.Add(newIndex);
	auto newSystem = gasWorld->getGas(newIndex);
	
	//GasSystemParams
	//TODO::
	//addGasSource数据可设置
	//make_vec3r(-1.2f, -1.0f, 0.0f)  0.4f  make_vec3r(1.0f, 0.0f, 0.0f)
	newSystem->addGasSource(make_vec3r(0.0f, -1.0f, 0.0f), 0.4f, make_vec3r(0.0f, 0.0f, 0.0f), 1.0f);
	
	//newSystem->addBox(make_vec3r(0, -0.7, 0), make_vec3r(0.8, 1.5, 0.8));

	//TODO::
	//setRenderData数据可设置
	gasWorld->setRenderData(newIndex, make_vec3r(255 / 255.0f, 255 / 255.0f, 255 / 255.0f), make_float3(0, -1, 0), 0.06f, 100);

    //Spawn Actor
    FActorSpawnParameters spawnParams;
	AUEGasSyetem* newGasSystem = GetWorld()->SpawnActor<AUEGasSyetem>(GasSystemClass,Location,FRotator::ZeroRotator, spawnParams);

	if (newGasSystem)
	{
		newGasSystem->InitGasSyetem(newSystem,newIndex);
		UEGasSystemsMap.Add(newIndex,newGasSystem);
		//ActiveGasSystems.Add(newGasSystem);
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("Failed to Spawn GasSystem Actor"));
	}
	return newGasSystem;
}

// [API 1] 创建GasSystem
void AGasManager::CreateGasSystem(GasSystemParams Params)
{
	if (!gasWorld)
	{
		UE_LOG(LogTemp, Warning, TEXT("gasWorld is nullpter"));
		return ;
	}

	//初始化 System
	int newIndex = gasWorld->initGasSystem(
		make_vec3r(0.0f), 
		Params.vorticity, 
		Params.diffusion, 
		Params.buoyancy, 
		Params.vcEpsilon, 
		Params.decreaseDensity
	);

	if (newIndex < 0) return ;

	//设置渲染参数
	auto newSystem = gasWorld->getGas(newIndex);
	//newSystem->addGasSource(make_vec3r(0.0f, -1.0f, 0.0f), 0.4f, make_vec3r(0.0f, 0.0f, 0.0f), 1.0f);
	gasWorld->setRenderData(
		newIndex,
		Params.color,
		Params.lightDir,
		Params.decay,
		Params.ambient
	);

	ActiveGasSystemIDs.Add(newIndex);
    
	UE_LOG(LogTemp, Warning, TEXT("Created Gas System ID: %d"), newIndex);
	return ;
}

// [API 2] 设置GasSystem的发射源
void AGasManager::AddGasSystemSource(int GasSystemID, GasSourceParams Params)
{
	//安全检查：确保这个 ID 是我们要管理的
	if (!ActiveGasSystemIDs.Contains(GasSystemID))
	{
		UE_LOG(LogTemp, Warning, TEXT("SpawnGasSystemActor 失败: GasSystemID %d 无效或未创建！"), GasSystemID);
		return ;
	}
	auto gasSystem = gasWorld->getGas(GasSystemID);
	gasSystem->addGasSource(
		Params.source,
		Params.radius,
		Params.velocity,
		Params.density
		);
}

// [API 3] 为已存在的系统生成渲染 Actor
AUEGasSyetem* AGasManager::SpawnGasSystemActor(int GasSystemID, FVector Location)
{
	//安全检查：确保这个 ID 是我们要管理的
	if (!ActiveGasSystemIDs.Contains(GasSystemID))
	{
		UE_LOG(LogTemp, Warning, TEXT("SpawnGasSystemActor 失败: GasSystemID %d 无效或未创建！"), GasSystemID);
		return nullptr;
	}
	// 检查是否已经有 Actor了
	if (UEGasSystemsMap.Contains(GasSystemID) && IsValid(UEGasSystemsMap[GasSystemID]))
	{
		return UEGasSystemsMap[GasSystemID];
	}
	
    //Spawn Actor
	auto gasSystem = gasWorld->getGas(GasSystemID);
	FActorSpawnParameters spawnParams;
	AUEGasSyetem* newGasSystem = GetWorld()->SpawnActor<AUEGasSyetem>(GasSystemClass,Location,FRotator::ZeroRotator, spawnParams);

	if (newGasSystem)
	{
		newGasSystem->InitGasSyetem(gasSystem,GasSystemID);
		UEGasSystemsMap.Add(GasSystemID,newGasSystem);
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("Failed to Spawn GasSystem Actor"));
	}
	return newGasSystem;
}

// [API 4] 设置GasSystem的颜色
void AGasManager::setGasSystemColor(int GasSystemID,vec3r gasColor)
{
	if (!ActiveGasSystemIDs.Contains(GasSystemID))
	{
		UE_LOG(LogTemp, Warning, TEXT("SpawnGasSystemActor 失败: GasSystemID %d 无效或未创建！"), GasSystemID);
		return ;
	}
	auto gasSystem = gasWorld->getGas(GasSystemID);
	gasSystem->setColor(gasColor);
	return ;
}

void AGasManager::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    if (gasWorld)
    {
	    // Run SDK simulation
    	for (auto gasID: ActiveGasSystemIDs)
    	{
    		gasWorld->update(gasID);
    	}
    }
}



void AGasManager::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	Super::EndPlay(EndPlayReason);
	if (gasWorld)
	{
		delete gasWorld;
		gasWorld = nullptr;
	}
}
