// FluidTestSubsystem.cpp
#include "FluidTestSubsystem.h"
#include "Engine/Engine.h"

#include "object/fluid_system.h"
#include "object/particle_fluid.h"
#include "cuda_viewer/cuda_viewer.h"
#include "object/config.h"
#include "common/timer.h"
#include "object/fluid_world.h"
using namespace physeng;
#define GRID_SIZE 64
uint numParticles = 0;
uint3 gridSize;

static void TestFluidPerformanceDemo()
{
	
	cudaInit(0, nullptr);

	FluidWorld* fluidWorld = new FluidWorld(make_vec3r(-15, 0, -15), make_vec3r(15, 25, 15));
	int fluidIndex = fluidWorld->initFluidSystem(make_vec3r(-4, 9, -4), make_vec3r(7, 10, 8) * 1.8, 0.0f, 0.05f);


	printf("Fluid Symbol:%d\n", fluidIndex);

	if (fluidIndex < 0) {
		exit(0);
	}
	printf("水体粒子数：%d\n", fluidWorld->getFluid(fluidIndex)->getCurNumParticles());

	fluidWorld->completeInit(fluidIndex);

	CudaViewer viewer;
	viewer.init(0, nullptr);
	viewer.bindFunctions();
	viewer.camera_trans[2] = -60;

	fluidWorld->initViewer(fluidIndex, viewer);
	//sdkCreateTimer(&timer);

	int frame = 0;

	viewer.updateCallback = [&]() {
		//sdkStartTimer(&timer);
		{
			// PHY_PROFILE("grid system update");
			fluidWorld->update(fluidIndex);
			frame++;
		}

		fluidWorld->updateViewer(fluidIndex, viewer);
		//sdkStopTimer(&timer);


		if (frame % 100 == 0) {
			//float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
			//LOG_OSTREAM_INFO << "frame " << frame << ", fps=" << ifps << ", time=" << sdkGetTimerValue(&timer) / 1000.f << std::endl;
			BENCHMARK_REPORT();
		}

		return true;
	};

	viewer.closeCallback = [&]() {
		//sdkDeleteTimer(&timer);
		delete fluidWorld;
		return true;
	};
	viewer.isPause = true;
	viewer.run();
	
	
}

void UFluidTestSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
	GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green, TEXT("=== 每次游戏运行都打印 ==="));
	UE_LOG(LogTemp, Warning, TEXT("FluidTestSubsystem::Initialize — 游戏实例启动！"));

	TestFluidPerformanceDemo();
	
}