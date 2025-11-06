// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "object/gas_world.h"
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

	/**
	 * 【新】在蓝图中设置的基础材质。
	 * 你应该将其设置为 M_ParticleBase (或你创建的任何材质)。
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Particle Manager|Config")
	UMaterialInterface* BaseMaterial;

private:
	/** 缓存当前实例的数量，用于检测变化 */
	int32 CurrentInstanceCount = 0;

	/** * 一个可重用的缓冲区，用于在 UpdateParticlePositions 中构建 FTransform 数组，
	 * 避免每帧都重新分配内存。
	 */
	TArray<FTransform> TransformBuffer;

	TArray<FVector> ParticlePositions;
	VecArray<vec3r, CPU> PositionHost;
};