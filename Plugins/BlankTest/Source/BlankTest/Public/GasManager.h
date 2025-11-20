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
    
	// [API] 动态创建并生成一个新的烟雾系统
	UFUNCTION(BlueprintCallable, Category = "Gas Simulation")
	AUEGasSyetem* CreateAndSpawnGasSystem(FVector Location);

	// [API 1] 仅创建GasSystem，不生成 Actor
	//UFUNCTION(BlueprintCallable, Category = "Gas Simulation")
	void CreateGasSystem(GasSystemParams Params);

	// [API 2] 设置GasSystem的发射源
	//UFUNCTION(BlueprintCallable, Category = "Gas Simulation")
	void AddGasSystemSource(int GasSystemID, GasSourceParams Params);

	// [API 3] 为已存在的系统生成渲染 Actor
	// 返回：生成的 Actor
	UFUNCTION(BlueprintCallable, Category = "Gas Simulation")
	AUEGasSyetem* SpawnGasSystemActor(int GasSystemID, FVector Location);

	// [API 4] 设置GasSystem的颜色
	UFUNCTION(BlueprintCallable, Category = "Gas Simulation")
	void setColorInUE(int GasSystemID, FVector Color)
	{
		setGasSystemColor(GasSystemID,make_vec3r(Color.X, Color.Y, Color.Z));
	}
	void setGasSystemColor(int GasSystemID, vec3r gasColor);

	//TODO:: 销毁逻辑

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

public: 
	// Called every frame (默认关闭)
	virtual void Tick(float DeltaTime) override;

private:
	
	GasWorld* gasWorld = nullptr;

	//维护所有已创建的 GasSystem ID
	UPROPERTY()
	TArray<int> ActiveGasSystemIDs;
	//Key: ID, Value: Actor
	UPROPERTY()
	TMap<int,AUEGasSyetem*> UEGasSystemsMap;

	//GasWorldParams
	GasWorldParams gasWorldParams;

	//GasSystemParams
	TArray<GasSystemParams> gasSystemParams;

};
