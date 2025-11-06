// Fill out your copyright notice in the Description page of Project Settings.

#include "ParticleManager.h"
#include "UObject/ConstructorHelpers.h"

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

    // --- 设置可配置属性的默认值 ---
    ParticleScale = FVector(0.2f);       // 默认缩放 (0.2, 0.2, 0.2)
    ParticleRotation = FRotator::ZeroRotator; // 默认无旋转

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

    // 假设我们希望 position 的分量在 0 到 10 之间
    float MinRange1 = 0.0f;
    float MaxRange1 = 1.0f;

    FVector position(
        FMath::FRandRange(MinRange1, MaxRange1), // 随机 X
        FMath::FRandRange(MinRange1, MaxRange1), // 随机 Y
        FMath::FRandRange(MinRange1, MaxRange1)  // 随机 Z
    );

    // 假设我们希望 position2 的分量在 20 到 100 之间
    float MinRange2 = 10.0f;
    float MaxRange2 = 11.0f;

    FVector position2(
        FMath::FRandRange(MinRange2, MaxRange2),
        FMath::FRandRange(MinRange2, MaxRange2),
        FMath::FRandRange(MinRange2, MaxRange2)
    );

    UpdateParticlePositions({position, position2});
}

void AParticleManager::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    UE_LOG(LogTemp, Warning, TEXT("AParticleManager::Tick is called"));
    // 假设我们希望 position 的分量在 0 到 10 之间
    float MinRange1 = 1.0f;
    float MaxRange1 = 10.0f;

    FVector position(
        FMath::FRandRange(MinRange1, MaxRange1), // 随机 X
        FMath::FRandRange(MinRange1, MaxRange1), // 随机 Y
        FMath::FRandRange(MinRange1, MaxRange1)  // 随机 Z
    );

    // 假设我们希望 position2 的分量在 20 到 100 之间
    float MinRange2 = 20.0f;
    float MaxRange2 = 30.0f;

    FVector position2(
        FMath::FRandRange(MinRange2, MaxRange2),
        FMath::FRandRange(MinRange2, MaxRange2),
        FMath::FRandRange(MinRange2, MaxRange2)
    );

    UpdateParticlePositions({position, position2});
}

void AParticleManager::ClearParticles()
{
    if (InstancedMeshComponent)
    {
        InstancedMeshComponent->ClearInstances();
    }
    CurrentInstanceCount = 0;
}

/**
 * 核心实现：只更新位置
 */
void AParticleManager::UpdateParticlePositions(const TArray<FVector>& NewPositions)
{
    if (!InstancedMeshComponent)
    {
        UE_LOG(LogTemp, Error, TEXT("InstancedMeshComponent IS NULL"));
        return;
    }
    UE_LOG(LogTemp, Warning, TEXT("AParticleManager::UpdateParticlePositions is called"));

    const int32 NewCount = NewPositions.Num();

    // 1. 准备 FTransform 数组
    // 我们重用 TransformBuffer，避免每帧分配新内存
    TransformBuffer.SetNumUninitialized(NewCount); // 设置数组大小，但不初始化元素

    // **从成员变量中获取固定的旋转和缩放**
    const FQuat RotationAsQuat = ParticleRotation.Quaternion();
    const FVector Scale = ParticleScale;

    // 2. 填充缓冲区 (只修改位置)
    for (int32 i = 0; i < NewCount; ++i)
    {
        TransformBuffer[i].SetComponents(RotationAsQuat, NewPositions[i], Scale);
    }

    // 3. 调用我们的“高级”更新函数来完成工作
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
        InstancedMeshComponent->BatchUpdateInstancesTransforms(0, NewTransforms, true /* bWorldSpace */, true /* bMarkRenderStateDirty */);
    }

    // 缓存新的数量
    CurrentInstanceCount = NewCount;
}