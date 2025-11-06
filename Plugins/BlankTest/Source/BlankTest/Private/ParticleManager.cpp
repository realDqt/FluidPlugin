// Fill out your copyright notice in the Description page of Project Settings.

#include "ParticleManager.h"
#include "UObject/ConstructorHelpers.h"
#include "object/fluid_system.h"
#include "object/particle_fluid.h"
#include "object/config.h"
#include "common/timer.h"
#include "object/fluid_world.h"


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


AParticleManager::AParticleManager()
{
    // 默认关闭 Tick
    PrimaryActorTick.bCanEverTick = true;

    // 创建根组件
    USceneComponent* Root = CreateDefaultSubobject<USceneComponent>(TEXT("RootComponent"));
    RootComponent = Root;

    // 【核心】创建 InstancedStaticMeshComponent
    InstancedMeshComponent = CreateDefaultSubobject<UInstancedStaticMeshComponent>(TEXT("InstancedMeshComponent"));
    InstancedMeshComponent->SetupAttachment(RootComponent);

    // 设置默认性能选项
    InstancedMeshComponent->SetMobility(EComponentMobility::Movable);
    InstancedMeshComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    InstancedMeshComponent->SetCastShadow(false);

    // （可选）加载默认网格体
    static ConstructorHelpers::FObjectFinder<UStaticMesh> SphereMesh(TEXT("/Engine/BasicShapes/Sphere"));
    if (SphereMesh.Succeeded())
    {
        InstancedMeshComponent->SetStaticMesh(SphereMesh.Object);
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("AParticleManager: Could not find default Sphere mesh! Please set one in the Blueprint."));
    }
}

void AParticleManager::BeginPlay()
{
    Super::BeginPlay();
    // 准备就绪，等待 UpdateParticlePositions 被调用
    

    TestFluidPerformanceDemo(0, nullptr);
    int numOfParticles = fluidWorld->getFluid(0)->getCurNumParticles();
    PositionHost = VecArray<vec3r, CPU>(numOfParticles);
    ParticlePositions.SetNumUninitialized(numOfParticles);

    // 检查 BaseMaterial 是否已在蓝图中设置
    if (BaseMaterial)
    {
        InstancedMeshComponent->SetMaterial(0, BaseMaterial);
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("AParticleManager: 'BaseMaterial' is not set in the Blueprint! Cannot create dynamic material."));
    }
}

void AParticleManager::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    if (fluidWorld)
    {
        fluidWorld->update(0); 
        // TODO: UE渲染
        auto fluid = fluidWorld->getFluid(0);
        check(fluid)
        auto& positionDevice = fluid->pf.getPositionRef();
        physeng::checkCudaError(cudaMemcpy(ParticlePositions.GetData(), positionDevice.m_data, PositionHost.size()*sizeof(vec3r), cudaMemcpyDeviceToHost));
        //copyArray<vec3r, MemType::CPU, MemType::GPU>(&PositionHost.m_data, &positionDevice.m_data, 0, PositionHost.size());

        check(PositionHost.size() == ParticlePositions.Num());
        UpdateParticlePositions(ParticlePositions);
        //UE_LOG(LogTemp, Warning, TEXT("fluidWorld != null, ParticlePositions[100].X = %f"), ParticlePositions[100].X); // 75276
    }
    else {
        UE_LOG(LogTemp, Warning, TEXT("fluidWorld == null"));
    }
    static int curFrame = 0;
    //UE_LOG(LogTemp, Warning, TEXT("Current Frame = %d"), curFrame++);
}

void AParticleManager::ClearParticles()
{
    if (InstancedMeshComponent)
    {
        InstancedMeshComponent->ClearInstances();
    }
    CurrentInstanceCount = 0;
}


void AParticleManager::UpdateParticlePositions(const TArray<FVector>& NewPositions)
{
    if (!InstancedMeshComponent)
    {
        return;
    }

    const int32 NewCount = NewPositions.Num();

    // 1. 处理 NewCount 为 0 的情况
    if (NewCount == 0)
    {
        if (CurrentInstanceCount > 0)
        {
            ClearParticles();
        }
        return;
    }

    // 2. 准备 FTransform 数组 (在主线程上调整大小)
    //    SetNumUninitialized (或 SetNum) 必须在主线程上调用
    TransformBuffer.SetNumUninitialized(NewCount);

    // 3. 准备固定的变换值 (这些将被并行任务捕获)
    const FQuat RotationAsQuat = FQuat::Identity; // 粒子本身的旋转
    float scale = 0.01;
    const FVector Scale = FVector(scale, scale, scale);

    // 4. 【新】使用 ParallelFor 并行填充缓冲区
    // ParallelFor 会自动将 NewCount 个任务分配到多个CPU核心
    ParallelFor(NewCount, [&](int32 i)
    {
        // 【高效旋转】
        const FVector& InPos = NewPositions[i];
        const FVector RotatedPos(InPos.X, -InPos.Z, InPos.Y);

        // 【填充缓冲区】
        // 索引 'i' 在每个并行任务中都是唯一的，所以写入 TransformBuffer[i] 是线程安全的。
        TransformBuffer[i].SetComponents(RotationAsQuat, RotatedPos, Scale);
    });
    // (此时，主线程会等待所有并行任务完成)

    // 5. 调用我们的“高级”更新函数来完成渲染提交
    // (UpdateParticleTransforms 内部包含了处理“生成”和“更新”的逻辑)
    UpdateParticleTransforms(TransformBuffer);
}


/**
 * 高级实现：更新完整的 Transform
 * (这个函数与上一个版本完全相同)
 */
void AParticleManager::UpdateParticleTransforms(const TArray<FTransform>& NewTransforms)
{
    if (!InstancedMeshComponent)
    {
        return;
    }

    const int32 NewCount = NewTransforms.Num();

    if (NewCount == 0)
    {
        // 如果新数量为 0，则清空
        if (CurrentInstanceCount > 0)
        {
            ClearParticles();
        }
        return;
    }

    // 检查粒子数量是否发生了变化
    if (NewCount != CurrentInstanceCount)
    {
        // **数量变化：这是“生成”步骤 (或重新生成)**
        // 清除旧的，然后批量添加新的
        InstancedMeshComponent->ClearInstances();
        InstancedMeshComponent->AddInstances(NewTransforms, false /* bShouldReturnIndices */);
    }
    else
    {
        // **数量未变：这是“更新”步骤**
        // 批量更新所有 Transform，这非常快
        //UE_LOG(LogTemp, Warning, TEXT("NewTransforms.Num = %d"), NewTransforms.Num());
        InstancedMeshComponent->BatchUpdateInstancesTransforms(0, NewTransforms, true /* bWorldSpace */, true /* bMarkRenderStateDirty */);
    }

    // 缓存新的数量
    CurrentInstanceCount = NewCount;
}