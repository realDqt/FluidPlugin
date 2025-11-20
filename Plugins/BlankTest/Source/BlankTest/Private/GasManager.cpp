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
	
	gasWorld = new GasWorld();
	if (!gasWorld)
	{
		printf("gasWorld is nullptr");
		UE_LOG(LogTemp, Warning, TEXT("Failed to create GasWorld instance"));
		return;
	}
	
	/*gasIndex = gasWorld->initGasSystem(make_vec3r(0.0f), 0.000002f, 0.000000f, 4.0f, 5.0f, 0.001f);
	if (gasIndex < 0)
	{
		UE_LOG(LogTemp, Error, TEXT("GasWorld SDK initGasSystem() 失败! 返回索引: %d"), gasIndex);
		return; // 安全地退出 BeginPlay，Tick 将不会运行
	}

	//UE_LOG(LogTemp, Log, TEXT("Gas Symbol:%d"), gasIndex);
	printf("Gas Symbol:%d\n", gasIndex);

	//TODO::
	//addGasSource数据可设置
	gasWorld->getGas(gasIndex)->addGasSource(make_vec3r(-1.2f, -1.0f, 0.0f), 0.5f, make_vec3r(1.0f, 0.0f, 0.0f), 1.0f);
	
	//gasWorld->getGas(gasIndex)->addBox(make_vec3r(0, -0.7, 0), make_vec3r(0.8, 1.5, 0.8));

	//TODO::
	//setRenderData数据可设置
	gasWorld->setRenderData(gasIndex, make_vec3r(255 / 255.0f, 255 / 255.0f, 255 / 255.0f), make_float3(0, -1, 0), 0.06f, 100);*/
}

AUEGasSyetem* AGasManager::SpawnGasSystem(FVector Location)
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
	auto newSystem = gasWorld->getGas(newIndex);
	
	//GasSystemParams
	//TODO::
	//addGasSource数据可设置
	newSystem->addGasSource(make_vec3r(-1.2f, -1.0f, 0.0f), 0.5f, make_vec3r(1.0f, 0.0f, 0.0f), 1.0f);
	
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
		ActiveGasSystems.Add(newGasSystem);
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("Failed to Spawn GasSystem Actor"));
	}
	return newGasSystem;
}

void AGasManager::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    if (gasWorld)
    {
	    // Run SDK simulation
    	for (AUEGasSyetem* gas: ActiveGasSystems)
    	{
    		if (gas)
    		{
    			gasWorld->update(gas->GetGasIndex());
    		}
    	}
    }
}

bool AGasManager::setGasSystemColor(int gasIndex,vec3r gasColor)
{
	if (gasWorld)
	{
		auto gasSystem = gasWorld->getGas(gasIndex);
		if (gasSystem)
		{
			gasColors[gasIndex] = gasColor;
			return true;
		}
	}
	return false;
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
