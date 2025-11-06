// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/InstancedStaticMeshComponent.h" // 包含 ISM 组件头文件
#include "ParticleActor.generated.h"

UCLASS()
class TESTPLUGIN_API AParticleActor : public AActor
{
	GENERATED_BODY()
    
public: 
	// Sets default values for this actor's properties
	AParticleActor();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public: 
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	/** * 实例化静态网格体组件。这将用于渲染所有粒子。
	 * 可以在蓝图子类中设置其 Static Mesh 和材质。
	 */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Particles")
	UInstancedStaticMeshComponent* InstancedMeshComponent;

	/** * 从外部（例如 Subsystem 或其他 Actor）调用此函数来更新所有实例的位置。
	 * 第一次调用时会生成实例，之后只会更新位置。
	 * @param NewPositions - 粒子世界坐标数组。
	 * @param ParticleScale - 粒子实例的均匀缩放大小（例如，如果你希望半径为 10cm，且你的网格体是 1m 大小，则传入 0.2）。
	 * @param bForceReinitialize - 如果为 true，即使粒子数量相同，也会重新创建所有实例。
	 */
	UFUNCTION(BlueprintCallable, Category = "Particles")
	void UpdateParticlePositions(const TArray<FVector>& NewPositions, float ParticleScale = 1.0f, bool bForceReinitialize = false);

private:
	/** 标记是否已经初始化过实例。 */
	bool bInstancesInitialized = false;

	/** 缓存当前的粒子数量，用于判断是否需要重新初始化 ISM。 */
	int32 CurrentParticleCount = 0;

	/** 临时数组，用于在更新时构建 FTransform。 */
	TArray<FTransform> TempTransforms;

};