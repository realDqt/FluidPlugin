// .h 文件
#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "Tickable.h" // 必须包含这个头文件
#include "object/gas_world.h"
#include "FluidTestSubsystem.generated.h"


UCLASS()
class UFluidTestSubsystem : public UGameInstanceSubsystem, public FTickableGameObject // 继承 FTickableGameObject
{
	GENERATED_BODY()
public:
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

	//~ Begin FTickableGameObject Interface
	/**
	 * 这是你的每帧执行函数
	 * @param DeltaTime - 距离上一帧的时间
	 */
	virtual void Tick(float DeltaTime) override;

	/**
	 * 必须实现这个函数，返回一个 Stat ID
	 */
	virtual TStatId GetStatId() const override;

	/**
	 * 控制是否允许 Tick
	 */
	virtual bool IsTickable() const override;
	//~ End FTickableGameObject Interface

	VecArray<vec3r, CPU> positionHost;

};