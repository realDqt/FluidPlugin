#pragma once

#include "CoreMinimal.h"
#include "Engine/Engine.h"
#include "object/gas_world.h"

#include "GameFramework/Actor.h"
#include "UObject/ConstructorHelpers.h"
#include "Components/StaticMeshComponent.h"
#include "Materials/MaterialInstanceDynamic.h"
//#include "Engine/VolumeTexture.h"
#include "GasManager.generated.h"

class UVolumnTexture;

UCLASS()
class BLANKTEST_API AGasManager : public AActor
{
	GENERATED_BODY()
    
public: 
	AGasManager();

protected:
	virtual void BeginPlay() override;


public: 
	// Called every frame (默认关闭)
	virtual void Tick(float DeltaTime) override;
	
	// 在UE编辑器中指定一个基础的体积材质 (我们将在第4步创建它)
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Smoke Simulation")
	class UMaterialInterface* BaseVolumeMaterial;

	// 用于显示烟雾体积的Cube
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Smoke Simulation")
	class UStaticMeshComponent* VolumeMeshComponent;

	// UE中的动态3D纹理对象
	UPROPERTY(Transient)
	class UVolumeTexture* SmokeVolumeTexture;

	// 动态材质实例 (MDI)，用于我们将纹理传递给着色器
	UPROPERTY(Transient)
	class UMaterialInstanceDynamic* DynamicVolumeMaterial;

private:

	/**
	 * Internal cache for the grid size, so we don't have to fetch it constantly.
	 */
	uint3 SDKGridSize;
	
};
