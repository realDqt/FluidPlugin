#include "NiagaraBasedOnCPU.h"
#include "Engine/Engine.h"
#include "UObject/ConstructorHelpers.h"
#include <cuda_runtime.h>
#include "object/fluid_system.h"
#include "object/particle_fluid.h"
#include "object/config.h"
#include "common/timer.h"
#include "object/fluid_world.h"

using namespace physeng;

uint numParticles2 = 0;
StopWatchInterface *timer2 = NULL;
static FluidWorld* fluidWorld2 = nullptr;

void TestFluidPerformanceDemo1(int argc, char** argv)
{
	cudaInit(argc, argv);

	fluidWorld2 = new FluidWorld(make_vec3r(-15, 0, -15), make_vec3r(15, 25, 15));
	int fluidIndex = fluidWorld2->initFluidSystem(make_vec3r(-4, 9, -4), make_vec3r(7, 10, 8) * 1.8, 0.0f, 0.05f);


	printf("Fluid Symbol:%d\n", fluidIndex);
    
	if (fluidIndex < 0) {
		exit(0);
	}
	printf("水体粒子数：%d\n", fluidWorld2->getFluid(fluidIndex)->getCurNumParticles());

	fluidWorld2->completeInit(fluidIndex);
    
}

// ----------------------------------------------------
// UE Actor 生命周期和模拟驱动
// ----------------------------------------------------

ANiagaraBasedOnCPU::ANiagaraBasedOnCPU()
{
    // 默认启用 Tick 来驱动模拟和数据回读
    PrimaryActorTick.bCanEverTick = true;

    // 创建根组件 (Actor 必须有根组件)
    USceneComponent* Root = CreateDefaultSubobject<USceneComponent>(TEXT("RootComponent"));
    RootComponent = Root;
    
    // TODO:这里通常还需要添加 UNiagaraComponent 组件来渲染粒子
    // 但我们将专注于数据流，渲染组件可以在蓝图中或 BeginPlay 中添加
}

void ANiagaraBasedOnCPU::BeginPlay()
{
    Super::BeginPlay();
    
    // 1. 初始化流体世界
    TestFluidPerformanceDemo1(0, nullptr);
    
    // 2. 预分配 CPU 数组空间
    if (fluidWorld2)
    {
        int numOfParticles = fluidWorld2->getFluid(0)->getCurNumParticles();
        
        // 确保 UE 数组和 SDK 辅助数组大小匹配
        PositionHost = VecArray<vec3r, CPU>(numOfParticles);
        ParticlePositions.SetNumUninitialized(numOfParticles);
        
        UE_LOG(LogTemp, Warning, TEXT("ANiagaraBasedOnCPU::BeginPlay - 预分配了 %d 个粒子内存。"), numOfParticles);
    }
}

void ANiagaraBasedOnCPU::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    if (fluidWorld2)
    {
        // 1. 运行 PBD 模拟 (GPU)
        fluidWorld2->update(0); 
        
        // 2. 获取 GPU 数据引用
        auto fluid = fluidWorld2->getFluid(0);
        check(fluid);
        auto& positionDevice = fluid->pf.getPositionRef();
        
        // 3. 【核心】GPU -> CPU 同步数据回读
        // 使用 CUDA API 将数据从 Device 复制到 Host
        physeng::checkCudaError(cudaMemcpy(ParticlePositions.GetData(), positionDevice.m_data, PositionHost.size()*sizeof(vec3r), cudaMemcpyDeviceToHost));
        
        
        // TODO: 此时，Niagara NDI (UNiagaraDataInterfaceCPUPBD) 可以在其 VMGetParticlePosition
        // 函数中安全地读取最新的 ParticlePositions 数据。
    }
    else {
        UE_LOG(LogTemp, Warning, TEXT("FluidWorld is null. Simulation not running."));
    }
}

void ANiagaraBasedOnCPU::ClearParticles()
{
    // 在 Niagara CPU 渲染方案中，此函数通常用于通知 Niagara Emitter 停止或重置。
    // 由于我们是直接驱动位置，这里可以简单地清空 CPU 数组或销毁粒子。
    ParticlePositions.Empty();
    UE_LOG(LogTemp, Warning, TEXT("ParticlePositions array cleared."));
}

// ----------------------------------------------------
// 警告/注意事项
// ----------------------------------------------------

/*
1. VecArray<vec3r, CPU> PositionHost 变量被保留但未被直接使用。
   您当前的 cudaMemcpy 已经跳过了它，直接将数据写入 ParticlePositions。这是可以接受的。

2. NDI 绑定：
   您还需要在蓝图或 C++ 中：
   a) 将一个 UNiagaraComponent 附加到此 Actor。
   b) 设置 Niagara System 模板。
   c) 将您自定义的 NDI (UNiagaraDataInterfaceCPUPBD) 的 Source Actor 属性绑定到此 ANiagaraBasedOnCPU 实例。
*/