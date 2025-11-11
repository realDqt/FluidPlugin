#pragma once

#include "CoreMinimal.h"
#include "Engine/Engine.h"
#include "object/gas_world.h"

#include "GameFramework/Actor.h"
#include "Components/InstancedStaticMeshComponent.h"
//#include "UObject/ConstructorHelpers.h"
//#include "Components/StaticMeshComponent.h"
//#include "Materials/MaterialInstanceDynamic.h"
//#include "Engine/VolumeTexture.h"
#include "GasTest.generated.h"

UCLASS()
class BLANKTEST_API AGasTest : public AActor
{
	GENERATED_BODY()
    
public: 
	AGasTest();

	//用于渲染粒子的 Instanced Mesh Component
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Gas Simulation")
	class UInstancedStaticMeshComponent* InstancedMeshComponent;

	//用于在蓝图中设置的基础材质
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Gas Simulation")
	class UMaterialInterface* BaseMaterial;

protected:
	virtual void BeginPlay() override;


public: 
	// Called every frame (默认关闭)
	virtual void Tick(float DeltaTime) override;


private:

	/**
	 * Internal cache for the grid size, so we don't have to fetch it constantly.
	 */
	//uint3 SDKGridSize;
	FIntVector SDKGridSize;

	//缓存模拟的世界边界和单元格大小，用于坐标转换
	FVector SimWorldMin;
	float SimCellLength;

	//粒子渲染的辅助函数和数据 (从 AParticleManager 复制)
	int32 CurrentInstanceCount = 0;
	TArray<FTransform> TransformBuffer;
	void ClearParticles();
	void UpdateParticlePositions(const TArray<FVector>& NewPositions);
	void UpdateParticleTransforms(const TArray<FTransform>& NewTransforms);
	
};

