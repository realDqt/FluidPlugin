// FluidTestSubsystem.cpp
#include "FluidTestSubsystem.h"
#include "Engine/Engine.h"


void UFluidTestSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);
	GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green, TEXT("=== 每次游戏运行都打印 ==="));
	UE_LOG(LogTemp, Warning, TEXT("FluidTestSubsystem::Initialize — 游戏实例启动！"));
	
}