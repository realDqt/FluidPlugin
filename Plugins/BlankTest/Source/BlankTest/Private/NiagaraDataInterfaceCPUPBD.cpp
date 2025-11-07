#include "NiagaraDataInterfaceCPUPBD.h"

#include <Context.h>

#include "NiagaraSystemInstance.h"

#include "NiagaraBasedOnCPU.h" 
#include "NiagaraTypes.h"
#include "VectorVM.h"
#include "NiagaraCustomVersion.h"



// ----------------------------------------------------
// NDI C++ 实现 (CPU/Game Thread)
// ----------------------------------------------------

// 辅助函数：安全获取 ANiagaraBasedOnCPU 指针
ANiagaraBasedOnCPU* UNiagaraDataInterfaceCPUPBD::GetNiagaraParticleManager(FNiagaraSystemInstance* SystemInstance) const
{
    if (!SystemInstance)
    {
        return nullptr;
    }
    // 使用 Cast 安全地将 SourceActor 转换为目标类型
    return Cast<ANiagaraBasedOnCPU>(SourceActor);
}

int32 UNiagaraDataInterfaceCPUPBD::PerInstanceDataSize() const
{
    return sizeof(FNiagaraPBD_CPUParticleData);
}

// 初始化 NDI 运行时数据实例
bool UNiagaraDataInterfaceCPUPBD::InitPerInstanceData(void* PerInstanceData, FNiagaraSystemInstance* SystemInstance)
{
    // 将 void* 转换为我们的自定义结构体指针
    FNiagaraPBD_CPUParticleData* InstanceData = new(PerInstanceData) FNiagaraPBD_CPUParticleData();
    
    ANiagaraBasedOnCPU* Manager = GetNiagaraParticleManager(SystemInstance);
    if (Manager)
    {
        // 【核心】缓存指向 ParticlePositions 数组的指针和数量
        InstanceData->PositionsPtr = &(Manager->ParticlePositions);
        InstanceData->NumParticles = Manager->GetParticleCount();
    }
    else
    {
        // 如果 Manager 无效，则保留空指针
        InstanceData->PositionsPtr = nullptr;
        InstanceData->NumParticles = 0;
        UE_LOG(LogTemp, Warning, TEXT("UNiagaraDataInterfaceCPUPBD: Source Actor not set or is not ANiagaraBasedOnCPU."));
    }
    
    return true;
}

// 暴露给 Niagara VM (CPU) 的函数签名
void UNiagaraDataInterfaceCPUPBD::GetFunctions(TArray<FNiagaraFunctionSignature>& OutFunctions)
{
    FNiagaraFunctionSignature Sig;
    Sig.Name = TEXT("GetParticlePosition");

    Sig.Inputs.Add(FNiagaraVariable(FNiagaraTypeDefinition::GetIntDef(), TEXT("ParticleIndex"))); // 输入粒子 ID
    Sig.Outputs.Add(FNiagaraVariable(FNiagaraTypeDefinition::GetVec3Def(), TEXT("Position"))); // 输出 FVector
    
    Sig.bMemberFunction = true; 
    Sig.bRequiresExecPin = false;
    OutFunctions.Add(Sig);
}

// 绑定函数：将 Niagara VM 调用映射到实际 C++ 函数
// 【UE 4.27 核心】使用 FVMExternalFunction
void UNiagaraDataInterfaceCPUPBD::GetVMExternalFunction(const FVMExternalFunctionBindingInfo& BindingInfo, void* InstanceData, FVMExternalFunction &OutFunc)
{
    if (BindingInfo.Name == TEXT("GetParticlePosition") && BindingInfo.GetNumInputs() == 1 && BindingInfo.GetNumOutputs() == 1)
    {
        // 使用 FVMExternalFunction::CreateUObject 进行绑定
        OutFunc = FVMExternalFunction::CreateUObject(this, &UNiagaraDataInterfaceCPUPBD::VMGetParticlePosition);
    }
}


// ----------------------------------------------------
// NDI 核心函数：在 CPU VM 中执行
// ----------------------------------------------------

void UNiagaraDataInterfaceCPUPBD::VMGetParticlePosition(FVectorVMContext& Context)
{
    // 1. 获取输入/输出指针
    VectorVM::FUserPtrHandler<int32> ParticleIndex(Context); 
    VectorVM::FExternalFuncRegisterHandler<FVector> OutPosition(Context); 

    // 2. 获取 ParticleManager 实例和数据
    FNiagaraPBD_CPUParticleData* InstanceData = (FNiagaraPBD_CPUParticleData*)Context.GetInstanceData();

    // 3. 检查缓存的数据是否有效
    if (!InstanceData || !InstanceData->PositionsPtr)
    {
        // 如果数据无效（例如 Actor 未设置），则快速跳过
        for (int32 i = 0; i < Context.GetNumInstances(); ++i)
        {
            *OutPosition.GetAndAdvance() = FVector::ZeroVector; 
        }
        return;
    }

    // 4. 从缓存的指针获取数据
    TArray<FVector>* Positions = InstanceData->PositionsPtr;
    const int32 NumParticles = InstanceData->NumParticles; // 使用缓存的数量

    // 5. 循环遍历当前批次的所有粒子
    for (int32 i = 0; i < Context.GetNumInstances(); ++i)
    {
        const int32 Index = ParticleIndex.GetAndAdvance(); 

        if (Index >= 0 && Index < NumParticles)
        {
            // 【修复】现在所有指针都已正确解析，赋值操作将成功
            *OutPosition.GetAndAdvance() = (*Positions)[Index]; 
        }
        else
        {
            // 索引越界
            *OutPosition.GetAndAdvance() = FVector::ZeroVector; 
        }
    }
}