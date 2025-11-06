// Fill out your copyright notice in the Description page of Project Settings.

#include "ParticleActor.h"
#include "UObject/ConstructorHelpers.h" // 用于加载默认资源

// Sets default values
AParticleActor::AParticleActor()
{
    // Set this actor to call Tick() every frame. You can turn this off to improve performance if you don't need it.
    PrimaryActorTick.bCanEverTick = false; // Actor 本身不需要每帧 Tick，我们通过外部调用 UpdateParticlePositions

    // 创建根组件，通常是一个 SceneComponent
    USceneComponent* Root = CreateDefaultSubobject<USceneComponent>(TEXT("RootComponent"));
    RootComponent = Root;

    // 创建 InstancedStaticMeshComponent
    InstancedMeshComponent = CreateDefaultSubobject<UInstancedStaticMeshComponent>(TEXT("InstancedMeshComponent"));
    InstancedMeshComponent->SetupAttachment(RootComponent);

    // 可选: 在 C++ 中设置一个默认的静态网格体。
    // 更推荐在蓝图子类中设置，因为设计师可以轻松更换。
    static ConstructorHelpers::FObjectFinder<UStaticMesh> SphereMesh(TEXT("/Engine/BasicShapes/Sphere"));
    if (SphereMesh.Succeeded())
    {
        InstancedMeshComponent->SetStaticMesh(SphereMesh.Object);
        // 设置默认材质（可选）
        // static ConstructorHelpers::FObjectFinder<UMaterial> DefaultMaterial(TEXT("/Engine/BasicShapes/BasicShapeMaterial"));
        // if (DefaultMaterial.Succeeded())
        // {
        //     InstancedMeshComponent->SetMaterial(0, DefaultMaterial.Object);
        // }
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("AParticleActor: Could not find default Sphere mesh! Please set one in blueprint."));
    }

    // 设置一些默认 ISM 属性，例如开启 Cast Shadows
    InstancedMeshComponent->SetCastShadow(false);
    // ... 其他你可能需要的属性
}

// Called when the game starts or when spawned
void AParticleActor::BeginPlay()
{
    Super::BeginPlay();
    
}

// Called every frame
void AParticleActor::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
}

void AParticleActor::UpdateParticlePositions(const TArray<FVector>& NewPositions, float ParticleScale, bool bForceReinitialize)
{
    if (!InstancedMeshComponent)
    {
        UE_LOG(LogTemp, Error, TEXT("AParticleActor: InstancedMeshComponent is null!"));
        return;
    }

    const int32 NewParticleCount = NewPositions.Num();

    // 检查是否需要重新初始化（第一次设置或粒子数量变化，或强制重新初始化）
    if (!bInstancesInitialized || NewParticleCount != CurrentParticleCount || bForceReinitialize)
    {
        // 清除所有现有实例
        InstancedMeshComponent->ClearInstances();

        if (NewParticleCount == 0)
        {
            bInstancesInitialized = false;
            CurrentParticleCount = 0;
            return; // 没有粒子，直接返回
        }

        // 预留空间，避免频繁的内存重新分配
        TempTransforms.SetNum(NewParticleCount);

        // 初始化所有实例的 Transform
        const FVector ScaleVector(ParticleScale);
        const FQuat Rotation = FQuat::Identity; // 粒子通常不需要旋转

        for (int32 i = 0; i < NewParticleCount; ++i)
        {
            TempTransforms[i] = FTransform(Rotation, NewPositions[i], ScaleVector);
        }

        // 批量添加所有实例
        InstancedMeshComponent->AddInstances(TempTransforms, false /*bWorldSpace*/);
        
        bInstancesInitialized = true;
        CurrentParticleCount = NewParticleCount;
    }
    else // 已经初始化过，且粒子数量没变，只需更新 Transform
    {
        if (NewParticleCount == 0)
        {
            // 如果粒子数量变为0，则清除
            InstancedMeshComponent->ClearInstances();
            bInstancesInitialized = false;
            CurrentParticleCount = 0;
            return;
        }

        // 预留空间
        TempTransforms.SetNum(NewParticleCount);

        const FVector ScaleVector(ParticleScale);
        const FQuat Rotation = FQuat::Identity;

        // 准备新的 Transforms
        for (int32 i = 0; i < NewParticleCount; ++i)
        {
            TempTransforms[i] = FTransform(Rotation, NewPositions[i], ScaleVector);
        }

        // 批量更新所有实例的 Transform
        // 注意：InstancedMeshComponent 和 HierarchicalInstancedStaticMeshComponent 的 BatchUpdateInstancesTransforms
        // 如果实例数量和更新数量不匹配，可能会有问题。
        // 最安全的方式是，如果数量变化，就 ClearInstances / AddInstances。
        // 如果数量不变，直接更新 Transform。
        InstancedMeshComponent->BatchUpdateInstancesTransforms(0, TempTransforms, false /*bWorldSpace*/, true /*bMarkRenderStateDirty*/);
    }
}