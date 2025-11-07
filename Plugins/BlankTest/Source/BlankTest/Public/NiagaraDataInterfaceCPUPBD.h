#pragma once

#include "CoreMinimal.h"
#include "NiagaraDataInterface.h"
#include "GameFramework/Actor.h"
#include "NiagaraCore.h" // 推荐，通常包含核心定义
#include "NiagaraTypes.h"
#include "NiagaraCommon.h"
#include "NiagaraShader.h"
#include "VectorVM.h"
#include "NiagaraDataInterfaceCPUPBD.generated.h"

class ANiagaraBasedOnCPU;

struct FNiagaraPBD_CPUParticleData
{
	// 缓存指向 CPU 粒子位置数组的指针
	TArray<FVector>* PositionsPtr;
	// 缓存粒子数量，避免每次都调用函数
	int32 NumParticles;

	// 构造函数
	FNiagaraPBD_CPUParticleData()
		: PositionsPtr(nullptr)
		, NumParticles(0)
	{}
};

// 继承自 UNiagaraDataInterface
UCLASS(EditInlineNew, Category = "PBD-CPU", meta = (DisplayName = "PBD Particle Data (CPU)"))
class BLANKTEST_API UNiagaraDataInterfaceCPUPBD : public UNiagaraDataInterface
{
	GENERATED_BODY()
	
public:
	// 【核心】在蓝图中设置，指向 NiagaraBasedOnCPU 实例
	UPROPERTY(EditAnywhere, Category = "Source Data")
	AActor* SourceActor;

	// ----------------------------------------------------
	// UNiagaraDataInterface 覆盖函数 (CPU/Game Thread)
	// ----------------------------------------------------

	virtual int32 PerInstanceDataSize() const override;

	// 创建 NDI 运行时数据实例
	virtual bool InitPerInstanceData(void* PerInstanceData, FNiagaraSystemInstance* SystemInstance) override;
    
	// 暴露给 Niagara VM (CPU) 的函数签名
	virtual void GetFunctions(TArray<FNiagaraFunctionSignature>& OutFunctions) override;

	// 绑定函数：将 HLSL 或 VM 函数调用映射到实际 C++ 函数
	virtual void GetVMExternalFunction(const FVMExternalFunctionBindingInfo& BindingInfo, void* InstanceData, FVMExternalFunction &OutFunc) override;

	// ----------------------------------------------------
	// 自定义 C++ 函数 (供 Niagara VM 内部调用)
	// ----------------------------------------------------
    
	// NDI 核心函数：用于获取给定索引的粒子位置
	/**
	 * @brief 核心函数。在 Niagara VM (CPU) 中执行，用于获取给定索引的粒子位置。
	 * @param Context - Niagara VM 的执行上下文。
	 * @param InstanceData - 由 Lambda 传入的、已初始化的 FNiagaraPBD_CPUParticleData 指针.
	 */
	
	//void VMGetParticlePosition(FVectorVMContext& Context);
	static void VMGetParticlePosition_Internal(FVectorVMContext& Context, FNiagaraPBD_CPUParticleData* InstanceData);
	
private:
	// 缓存 NiagaraBasedOnCPU 的指针
	ANiagaraBasedOnCPU* GetNiagaraParticleManager(FNiagaraSystemInstance* SystemInstance) const;
};
