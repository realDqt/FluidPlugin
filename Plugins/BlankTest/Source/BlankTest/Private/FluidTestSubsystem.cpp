// FluidTestSubsystem.cpp
#include "FluidTestSubsystem.h"
#include "Engine/Engine.h"

#include "object/gas_system.h"
#include "object/grid_gas.h"
#include "object/config.h"
#include "common/timer.h"

#include "Stats/Stats.h"

using namespace physeng;

uint numParticles = 0;
//uint3 gridSize;

StopWatchInterface *timer = NULL;

void TestFluidPerformanceDemo(int argc, char** argv, GasWorld*& gasWorld) {
	int scene = 0;

	cudaInit(argc, argv);

	gasWorld = new GasWorld();

	check(gasWorld != nullptr);

	int gasIndex = gasWorld->initGasSystem(make_vec3r(0.0f), 0.000002f, 0.000000f, 4.0f, 5.0f, 0.001f);

	//printf("气体标识：%d\n", gasIndex);
	if(gasIndex < 0){
		exit(0);
	}

	gasWorld->getGas(gasIndex)->addGasSource(make_vec3r(-1.2f, -1.0f, 0.0f), 0.5f, make_vec3r(1.0f, 0.0f, 0.0f), 1.0f);
	gasWorld->getGas(gasIndex)->addBox(make_vec3r(0, -0.7, 0), make_vec3r(0.8, 1.5, 0.8));
	gasWorld->setRenderData(gasIndex, make_vec3r(255 / 255.0f, 255 / 255.0f, 255 / 255.0f), make_float3(0, -1, 0), 0.06f, 100);

	/*
	for (int i = 0; i < 10; ++i)
	{
		gasWorld->update(0);
		UE_LOG(LogTemp, Warning, TEXT("end smi for %d"), i);
	}
	*/
}

void UFluidTestSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
	GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green, TEXT("=== 每次游戏运行都打印 ==="));
	UE_LOG(LogTemp, Warning, TEXT("FluidTestSubsystem::Initialize — 游戏实例启动！"));

	TestFluidPerformanceDemo(0, nullptr, gasWorld);
	
}

bool UFluidTestSubsystem::IsTickable() const
{
	return true;
}

void UFluidTestSubsystem::Tick(float DeltaTime)
{
	if (gasWorld)
	{
		gasWorld->update(0); //这行导致crash
		UE_LOG(LogTemp, Warning, TEXT("gasWorld != null"));
	}
	else {
		UE_LOG(LogTemp, Warning, TEXT("gasWorld == null"));
	}
	static int curFrame = 0;
	UE_LOG(LogTemp, Warning, TEXT("Current Frame = %d"), curFrame++);
}

void UFluidTestSubsystem::Deinitialize()
{
	
}

// 名字随意取，只要全局唯一即可
DECLARE_CYCLE_STAT(TEXT("FluidTestSubsystem_Tick"), STAT_FluidTestSubsystem_Tick, STATGROUP_Game);



TStatId UFluidTestSubsystem::GetStatId() const
{
	// 直接把 STAT 对象返回即可
	return GET_STATID(STAT_FluidTestSubsystem_Tick);
}