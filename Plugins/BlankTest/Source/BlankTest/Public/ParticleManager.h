// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "ParticleManager.generated.h"

UCLASS()
class BLANKTEST_API AParticleManager : public AActor
{
	GENERATED_BODY()
    
public: 
	AParticleManager();

protected:
	virtual void BeginPlay() override;

	/**
	 * 【核心】实例化静态网格体组件。
	 * 这一个组件将负责渲染所有的粒子（小球）。
	 */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
	UInstancedStaticMeshComponent* InstancedMeshComponent;

public: 
	// Called every frame (默认关闭)
	virtual void Tick(float DeltaTime) override;

	// --- 可配置的固定属性 ---

	/** * 在编辑器中设置的粒子固定缩放。
	 * 这将在所有实例之间共享。
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Particle Manager|Config")
	FVector ParticleScale;

	/** * 在编辑器中设置的粒子固定旋转。
	 * 这将在所有实例之间共享。
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Particle Manager|Config")
	FRotator ParticleRotation;


	// --- 控制函数 ---

	/**
	 * 【核心更新函数】用于“生成”或“更新”所有粒子的位置。
	 * 它将自动使用在蓝图中设置的 ParticleScale 和 ParticleRotation。
	 *
	 * @param NewPositions - 所有粒子的新世界坐标数组。
	 */
	UFUNCTION(BlueprintCallable, Category = "Particle Manager")
	void UpdateParticlePositions(const TArray<FVector>& NewPositions);

	/**
	 * （高级功能）如果你需要完全控制，此函数仍然可用。
	 * @param NewTransforms - 所有粒子的新 Transform 数组。
	 */
	UFUNCTION(BlueprintCallable, Category = "Particle Manager")
	void UpdateParticleTransforms(const TArray<FTransform>& NewTransforms);

	/**
	 * 清除所有粒子实例。
	 */
	UFUNCTION(BlueprintCallable, Category = "Particle Manager")
	void ClearParticles();

private:
	/** 缓存当前实例的数量，用于检测变化 */
	int32 CurrentInstanceCount = 0;

	/** * 一个可重用的缓冲区，用于在 UpdateParticlePositions 中构建 FTransform 数组，
	 * 避免每帧都重新分配内存。
	 */
	TArray<FTransform> TransformBuffer;
};