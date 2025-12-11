// Fill out your copyright notice in the Description page of Project Settings.

#include "ParticleManager.h"

#include "NavigationSystemTypes.h"
#include "UObject/ConstructorHelpers.h"
#include "object/fluid_system.h"
#include "object/particle_fluid.h"
#include "object/config.h"
#include "common/timer.h"
#include "object/fluid_world.h"


using namespace physeng;

static FluidWorld* fluidWorld = nullptr;
static FVector OriSDKPos = FVector(-4, 9, -4);
static FVector DeltaSDKPos = FVector(0, 0, 0);
static int frame = 0;

static int FluidParticleCount = 0;
static int RigidOrSandParticleCount = 0;

static void InitFluidPerformanceDemo(int argc, char** argv)
{
    cudaInit(argc, argv);

    fluidWorld = new FluidWorld(make_vec3r(-15, 0, -15), make_vec3r(15, 25, 15));
    int fluidIndex = fluidWorld->initFluidSystem(make_vec3r(OriSDKPos.X, OriSDKPos.Y, OriSDKPos.Z), make_vec3r(7, 10, 8) * 1.8, 0.0f, 0.05f);

    printf("Fluid Symbol:%d\n", fluidIndex);
    
    if (fluidIndex < 0) {
        exit(0);
    }
    FluidParticleCount = fluidWorld->getFluid(fluidIndex)->getCurNumParticles();
    RigidOrSandParticleCount = 0;

    fluidWorld->completeInit(fluidIndex);
    
}

static void UpdateFluidPerformanceDemo()
{
    fluidWorld->update(0);
    frame++;
}

static void InitRigidFloatDemo(int argc, char** argv)
{
    cudaInit(argc, argv);
    vec3r worldMin = make_vec3r(-15, 0, -15), worldMax = make_vec3r(15, 25, 15);
    fluidWorld = new FluidWorld(worldMin, worldMax);
    int fluidIndex = fluidWorld->initFluidSystem(make_vec3r(-4, 9, 0), make_vec3r(20, 18, 20), 0.0f, 0.05f, true);

    FluidParticleCount = fluidWorld->getFluid(fluidIndex)->getCurNumParticles();
    fluidWorld->addCube(make_vec3r(10, 10, -5), make_vec3r(4.0));
    fluidWorld->addCube(make_vec3r(10, 10, 5), make_vec3r(4.0));
    if (fluidIndex < 0) {
        exit(0);
    }
    RigidOrSandParticleCount = fluidWorld->getFluid(fluidIndex)->getCurNumParticles() - FluidParticleCount;
    fluidWorld->completeInit(fluidIndex);
}

static void UpdateRigidFloatDemo()
{
    vec3r worldMin = make_vec3r(-15, 0, -15), worldMax = make_vec3r(15, 25, 15);
    fluidWorld->setWorldBoundary(worldMin - make_vec3r(6 * sinr(frame * fluidWorld->getDt()), 0, 0), worldMax);
    fluidWorld->update(0);
    frame++;
}

static void InitKD(EFluidDemoType FluidDemoType)
{
    switch (FluidDemoType)
    {
        case EFluidDemoType::PERFORMANCE:
            InitFluidPerformanceDemo(0, nullptr);
            break;
        case EFluidDemoType::RIGID_FLOAT:
            InitRigidFloatDemo(0, nullptr);
            break;
        default:
            break;
    }

    std::cout << "Fluid: " << FluidParticleCount << " RigidOrSand: " << RigidOrSandParticleCount << std::endl;
}

static void UpdateKD(EFluidDemoType FluidDemoType)
{
    switch (FluidDemoType)
    {
        case EFluidDemoType::PERFORMANCE:
            UpdateFluidPerformanceDemo();
            break;
        case EFluidDemoType::RIGID_FLOAT:
            UpdateRigidFloatDemo();
            break;
        default:
            break;
    }
}



AParticleManager::AParticleManager()
{
    // 默认关闭 Tick
    PrimaryActorTick.bCanEverTick = true;

    // 创建根组件
    USceneComponent* Root = CreateDefaultSubobject<USceneComponent>(TEXT("RootComponent"));
    RootComponent = Root;


    InstancedMeshComponentFluid = CreateDefaultSubobject<UInstancedStaticMeshComponent>(TEXT("InstancedMeshComponent"));
    InstancedMeshComponentFluid->SetupAttachment(RootComponent);
    InstancedMeshComponentFluid->SetMobility(EComponentMobility::Movable);
    InstancedMeshComponentFluid->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    InstancedMeshComponentFluid->SetCastShadow(false);

    InstancedMeshComponentRigidOrSand = CreateDefaultSubobject<UInstancedStaticMeshComponent>(TEXT("InstancedMeshComponent2"));
    InstancedMeshComponentRigidOrSand->SetupAttachment(RootComponent);
    InstancedMeshComponentRigidOrSand->SetMobility(EComponentMobility::Movable);
    InstancedMeshComponentRigidOrSand->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    InstancedMeshComponentRigidOrSand->SetCastShadow(false);

    // （可选）加载默认网格体
    static ConstructorHelpers::FObjectFinder<UStaticMesh> SphereMesh(TEXT("/Engine/BasicShapes/Sphere"));
    if (SphereMesh.Succeeded())
    {
        InstancedMeshComponentFluid->SetStaticMesh(SphereMesh.Object);
        // 检查 BaseMaterial 是否已在蓝图中设置
        if (BaseMaterialFluid)
        {
            InstancedMeshComponentFluid->SetMaterial(0, BaseMaterialFluid);
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("AParticleManager: 'BaseMaterial' is not set in the Blueprint! Cannot create dynamic material."));
        }
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
}

static TArray<FString> ParseStringByPipe(const FString& InputStr)
{
    TArray<FString> OutArray;
    
    // ParseIntoArray 参数说明：
    // 1. OutArray: 接收结果的数组引用
    // 2. TEXT("|"): 分割符
    // 3. true: CullEmpty (剔除空项)。如果设为 true，像 "a||b" 这种会有中间空字符串的情况会被忽略，直接变成 ["a", "b"]。
    InputStr.ParseIntoArray(OutArray, TEXT("|"), true);
    
    return OutArray;
}

void AParticleManager::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    
    // ---------------------------cmd test begin: --------------------------------
    APlayerController* PC = GetWorld()->GetFirstPlayerController();
    if (PC && PC->IsInputKeyDown(EKeys::SpaceBar))
    {
        ProcessCmd(ParseStringByPipe(TEXT("fluid|create")));
    }

    if (PC && PC->IsInputKeyDown(EKeys::C))
    {
        ProcessCmd(ParseStringByPipe(TEXT("fluid|system|color|1.00|0.00|0.00")));
    }
    
    if (PC && PC->IsInputKeyDown(EKeys::P))
    {
        ProcessCmd(ParseStringByPipe(TEXT("fluid|system|position|20.00|20.00|20.00")));
    }
    // ---------------------------cmd test end: --------------------------------
    
    
    if (fluidWorld)
    {
        UpdateKD(CurDemoType);
        auto fluid = fluidWorld->getFluid(0);
        check(fluid)
        auto& positionDevice = fluid->pf.getPositionRef();
        physeng::checkCudaError(cudaMemcpy(ParticlePositions.GetData(), positionDevice.m_data, ParticlePositions.Num()*sizeof(vec3r), cudaMemcpyDeviceToHost));
        UpdateParticlePositions(ParticlePositions);
    }
    else {
        UE_LOG(LogTemp, Warning, TEXT("fluidWorld == null"));
    }
}

void AParticleManager::ClearParticles()
{
    if (InstancedMeshComponentFluid)
    {
        InstancedMeshComponentFluid->ClearInstances();
    }
    CurrentInstanceCount = 0;
}


void AParticleManager::UpdateParticlePositions(const TArray<FVector>& NewPositions)
{
    if (!InstancedMeshComponentFluid)
    {
        return;
    }

    const int32 NewCount = NewPositions.Num();
    
    if (NewCount == 0)
    {
        if (CurrentInstanceCount > 0)
        {
            ClearParticles();
        }
        return;
    }
    
    FluidTransformBuffer.SetNumUninitialized(FluidParticleCount);
    RigidOrSandTransformBuffer.SetNumUninitialized(RigidOrSandParticleCount);
    check(NewCount == FluidParticleCount + RigidOrSandParticleCount);
    
    const FQuat RotationAsQuat = FQuat::Identity; // 粒子本身的旋转
    float scale = 0.005f; // scale = 0.005
    const FVector Scale = FVector(scale, scale, scale);
    
    ParallelFor(NewCount, [&](int32 i)
    {
        const FVector& InPos = NewPositions[i] + DeltaSDKPos;
        const FVector RotatedPos = CoordsSDK2UE(InPos);
        
        if (i < FluidParticleCount)
        {
            FluidTransformBuffer[i].SetComponents(RotationAsQuat, RotatedPos, Scale);
        }else
        {
            RigidOrSandTransformBuffer[i - FluidParticleCount].SetComponents(RotationAsQuat, RotatedPos, Scale);
        }
    });
    // (此时，主线程会等待所有并行任务完成)
    
    UpdateParticleTransforms(FluidTransformBuffer);
}

void AParticleManager::UpdateParticleTransforms(const TArray<FTransform>& NewTransforms)
{
    if (!InstancedMeshComponentFluid)
    {
        return;
    }

    const int32 NewCount = NewTransforms.Num();

    if (NewCount == 0)
    {
        if (CurrentInstanceCount > 0)
        {
            ClearParticles();
        }
        return;
    }
    
    if (NewCount != CurrentInstanceCount)
    {
        InstancedMeshComponentFluid->ClearInstances();
        InstancedMeshComponentFluid->AddInstances(NewTransforms, false /* bShouldReturnIndices */);
    }
    else
    {
        InstancedMeshComponentFluid->BatchUpdateInstancesTransforms(0, NewTransforms, true /* bWorldSpace */, true /* bMarkRenderStateDirty */);
    }
    CurrentInstanceCount = NewCount;
}



void AParticleManager::ProcessCmd(const TArray<FString>& cmdList)
{
    if (cmdList[1].Contains("create"))
    {
        CreateFluidSystem(EFluidDemoType::RIGID_FLOAT);
    }else if (cmdList[1].Contains("system"))
    {
        if (cmdList[2].Contains("position"))
        {
            SetFluidSystemPos(cmdList);
        }else if (cmdList[2].Contains("color"))
        {
            SetFluidSystemColor(cmdList);
        }
    }
}

void AParticleManager::SetFluidSystemColor(const FVector& Color)
{
    DynamicVolumeMaterialFluid = UMaterialInstanceDynamic::Create(BaseMaterialFluid, this);
    if (DynamicVolumeMaterialFluid)
    {
        DynamicVolumeMaterialFluid->SetVectorParameterValue(FName("BaseColor"), FLinearColor(Color.X, Color.Y, Color.Z));
        InstancedMeshComponentFluid->SetMaterial(0, DynamicVolumeMaterialFluid);
    }
}

void AParticleManager::SetFluidSystemPos(const FVector& UEPosition)
{
    const FVector SDKPos = CoordsUE2SDK(UEPosition);
    DeltaSDKPos = SDKPos - OriSDKPos;
}

void AParticleManager::CreateFluidSystem(EFluidDemoType DemoType)
{
    InitKD(DemoType);
    CurDemoType = DemoType;
    
    int numOfParticles = fluidWorld->getFluid(0)->getCurNumParticles();
    ParticlePositions.SetNumUninitialized(numOfParticles);
}


static FVector StrArr2FVec(const TArray<FString>& StrArr)
{
    // 1. 安全检查：确保数组至少有3个元素，否则访问会导致 crash
    int32 Num = StrArr.Num();
    if (Num < 3)
    {
        // 如果数据不足，建议输出一条日志（可选），并返回零向量
        UE_LOG(LogTemp, Error, TEXT("StrArr2FVec: Array length is less than 3!"));
        return FVector::ZeroVector;
    }

    // 2. 提取最后三个元素并转换为 float
    // Num - 3 对应 X
    // Num - 2 对应 Y
    // Num - 1 对应 Z
    float X = FCString::Atof(*StrArr[Num - 3]);
    float Y = FCString::Atof(*StrArr[Num - 2]);
    float Z = FCString::Atof(*StrArr[Num - 1]);

    // 3. 构造并返回
    return FVector(X, Y, Z);
}

void AParticleManager::SetFluidSystemPos(const TArray<FString>& cmdList)
{
    FVector UEPos = StrArr2FVec(cmdList);
    SetFluidSystemPos(UEPos);
}

void AParticleManager::SetFluidSystemColor(const TArray<FString>& cmdList)
{
    FVector Color = StrArr2FVec(cmdList);
    SetFluidSystemColor(Color);
}

FVector AParticleManager::CoordsSDK2UE(const FVector& SDKPosition)
{
    FVector UEPosition = FVector(SDKPosition.X, -SDKPosition.Z, SDKPosition.Y);
    return UEPosition;
}

FVector AParticleManager::CoordsUE2SDK(const FVector& UEPosition)
{
    FVector SDKPosition = FVector(UEPosition.X, UEPosition.Z, -UEPosition.Y);
    return SDKPosition;
}



