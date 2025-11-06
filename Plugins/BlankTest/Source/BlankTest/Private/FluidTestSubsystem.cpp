// FluidTestSubsystem.cpp
#include "FluidTestSubsystem.h"
#include "Engine/Engine.h"
/*
#include "object/fluid_system.h"
#include "object/particle_fluid.h"
#include "object/config.h"
#include "common/timer.h"
#include "object/fluid_world.h"
#include "Engine/World.h"
#include "Stats/Stats.h"

using namespace physeng;

uint numParticles = 0;
//uint3 gridSize;

StopWatchInterface *timer = NULL;

static FluidWorld* fluidWorld = nullptr;
void TestFluidPerformanceDemo(int argc, char** argv)
{
	cudaInit(argc, argv);

	fluidWorld = new FluidWorld(make_vec3r(-15, 0, -15), make_vec3r(15, 25, 15));
	int fluidIndex = fluidWorld->initFluidSystem(make_vec3r(-4, 9, -4), make_vec3r(7, 10, 8) * 1.8, 0.0f, 0.05f);


	printf("Fluid Symbol:%d\n", fluidIndex);
    
	if (fluidIndex < 0) {
		exit(0);
	}
	printf("水体粒子数：%d\n", fluidWorld->getFluid(fluidIndex)->getCurNumParticles());

	fluidWorld->completeInit(fluidIndex);
}

void UFluidTestSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
	GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green, TEXT("=== 每次游戏运行都打印 ==="));
	UE_LOG(LogTemp, Warning, TEXT("FluidTestSubsystem::Initialize — 游戏实例启动！"));

	TestFluidPerformanceDemo(0, nullptr);
	positionHost = VecArray<vec3r, CPU>(fluidWorld->getFluid(0)->getCurNumParticles());
}

bool UFluidTestSubsystem::IsTickable() const
{
	return true;
}

void UFluidTestSubsystem::Tick(float DeltaTime)
{
	if (fluidWorld)
	{
		fluidWorld->update(0); 
		// TODO: UE渲染
		auto fluid = fluidWorld->getFluid(0);
		check(fluid)
		auto& positionDevice = fluid->pf.getPositionRef();
		copyArray<vec3r, MemType::CPU, MemType::GPU>(&positionHost.m_data, &positionDevice.m_data, 0, positionHost.size());
		//UE_LOG(LogTemp, Warning, TEXT("fluidWorld != null, positionDevice.size() = %d"), positionDevice.size()); // 75276
	}
	else {
		UE_LOG(LogTemp, Warning, TEXT("fluidWorld == null"));
	}
	static int curFrame = 0;
	//UE_LOG(LogTemp, Warning, TEXT("Current Frame = %d"), curFrame++);
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
*/