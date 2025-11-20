#pragma once

#include "CoreMinimal.h"
#include "Engine/Engine.h"
#include "object/gas_world.h"
#include "GasUtils.h"
#include "common/cuda/nvVector.h"

#include "GameFramework/Actor.h"
#include "UObject/ConstructorHelpers.h"
#include "Components/StaticMeshComponent.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "UEGasSyetem.generated.h"

class UVolumeTexture;;
//struct FGasSyetemParams;

UCLASS()
class BLANKTEST_API AUEGasSyetem : public AActor
{
	GENERATED_BODY()
public:
	AUEGasSyetem();

	void InitGasSyetem(GasSystem* InGasSystem, int InGasIndex);
	
	UFUNCTION(BlueprintCallable, Category = "Gas")
	int GetGasIndex() const { return GasIndex; }
	
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
	GasSystem* GasSystemPtr = nullptr;
	int GasIndex = -1;
	uint3 SDKGridSize;

	//Params
	GasSystemParams gasSystemParams;
};
