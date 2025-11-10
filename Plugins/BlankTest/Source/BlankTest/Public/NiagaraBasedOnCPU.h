// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/InstancedStaticMeshComponent.h"
#include "object/gas_world.h"
#include "NiagaraBasedOnCPU.generated.h"

UCLASS()
class BLANKTEST_API ANiagaraBasedOnCPU : public AActor
{
	GENERATED_BODY()
    
public: 
	ANiagaraBasedOnCPU();

protected:
	virtual void BeginPlay() override;


public: 
	// Called every frame (默认关闭)
	virtual void Tick(float DeltaTime) override;

	// 【新增】用于保护 ParticlePositions 访问的互斥锁
	FCriticalSection DataLock;

	// 【新增】供 NDI 读取的 CPU 粒子位置数组 (必须是 Public/BlueprintReadOnly)
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Particle Manager|Data")
	TArray<FVector> ParticlePositions; // 保持这个数组作为CPU数据源

	// 【新增】供 NDI 获取粒子数量的函数
	UFUNCTION(BlueprintCallable, Category = "Particle Manager|Data")
	int32 GetParticleCount() const { return ParticlePositions.Num(); }

	/**
	 * 清除所有粒子实例。
	 */
	UFUNCTION(BlueprintCallable, Category = "Particle Manager|Data")
	void ClearParticles();


private:
	VecArray<vec3r, CPU> PositionHost;
};