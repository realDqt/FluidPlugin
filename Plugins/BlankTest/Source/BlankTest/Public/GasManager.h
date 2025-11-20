#pragma once

#include "CoreMinimal.h"
#include "Engine/Engine.h"
#include "object/gas_world.h"
#include "GasUtils.h"
#include "GameFramework/Actor.h"
#include "UEGasSyetem.h"
#include "GasManager.generated.h"

//class UVolumeTexture;;

UCLASS()
class BLANKTEST_API AGasManager : public AActor
{
	GENERATED_BODY()
    
public: 
	AGasManager();
	// [API] 蓝图可调用的生成函数
	UPROPERTY(EditDefaultsOnly, Category = "Configuration")
	TSubclassOf<AUEGasSyetem> GasSystemClass; 
    
	// [API] 动态生成一个新的烟雾系统
	UFUNCTION(BlueprintCallable, Category = "Gas Simulation")
	AUEGasSyetem* SpawnGasSystem(FVector Location);

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

public: 
	// Called every frame (默认关闭)
	virtual void Tick(float DeltaTime) override;
	

	bool setGasSystemColor(int gasIndex, vec3r gasColor);

private:
	
	GasWorld* gasWorld = nullptr;
	
	// 存储所有生成的 GasSystem 引用
	UPROPERTY()
	TArray<AUEGasSyetem*> ActiveGasSystems;

	//GasParams
	GasWorldParams gasWorldParams;
	std::vector<vec3r> gasColors;
};
