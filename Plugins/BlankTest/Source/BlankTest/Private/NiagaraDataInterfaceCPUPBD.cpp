#include "NiagaraDataInterfaceCPUPBD.h"

#include "NiagaraSystemInstance.h"
#include "NiagaraBasedOnCPU.h" 
#include "NiagaraTypes.h"
#include "VectorVM.h"
#include "NiagaraCustomVersion.h"
#include "NiagaraConstants.h"



// ----------------------------------------------------
// NDI C++ 实现 (CPU/Game Thread)
// ----------------------------------------------------

void UNiagaraDataInterfaceCPUPBD::PostInitProperties()
{
    Super::PostInitProperties();

    if (HasAnyFlags(RF_ClassDefaultObject))
    {
        ENiagaraTypeRegistryFlags Flags = ENiagaraTypeRegistryFlags::AllowAnyVariable | ENiagaraTypeRegistryFlags::AllowParameter;
        FNiagaraTypeRegistry::Register(FNiagaraTypeDefinition(GetClass()), Flags);
    }
}

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
        //OutFunc = FVMExternalFunction::CreateUObject(this, &UNiagaraDataInterfaceCPUPBD::VMGetParticlePosition);
        // 1. 将 void* 转换为我们自己的数据类型
        FNiagaraPBD_CPUParticleData* MyInstanceData = (FNiagaraPBD_CPUParticleData*)InstanceData;

        // 2. 【核心】使用 CreateLambda 捕获 MyInstanceData 指针
        OutFunc = FVMExternalFunction::CreateLambda(
            [MyInstanceData](FVectorVMContext& Context)
            {
                // 3. 调用我们的静态函数，并将 Context 和捕获的 InstanceData 传递进去
                UNiagaraDataInterfaceCPUPBD::VMGetParticlePosition_Internal(Context, MyInstanceData);
            }
        );
    }
}


// ----------------------------------------------------
// NDI 核心函数：在 CPU VM 中执行
// ----------------------------------------------------

void UNiagaraDataInterfaceCPUPBD::VMGetParticlePosition_Internal(FVectorVMContext& Context,FNiagaraPBD_CPUParticleData* InstanceData)
{
    // 1. 获取输入/输出指针
    VectorVM::FExternalFuncInputHandler<int32> ParticleIndexHandler(Context);
    VectorVM::FExternalFuncRegisterHandler<FVector> OutPositionHandler(Context);
    
    // 2. 检查缓存的数据是否有效 (现在检查的是 PositionsPtr)
    if (!InstanceData || !InstanceData->PositionsPtr)
    {
        // 如果数据无效，快速将所有输出设置为零并返回
        for (int32 i = 0; i < Context.GetNumInstances(); ++i)
        {
            ParticleIndexHandler.GetAndAdvance(); // 必须消耗输入
            *(OutPositionHandler.GetDestAndAdvance()) = FVector::ZeroVector; 
        }
        return;
    }

    // 3. 从缓存的指针获取数据 (非常快)
    const TArray<FVector>& Positions = *(InstanceData->PositionsPtr); 
    const int32 NumParticles = InstanceData->NumParticles; // 使用缓存的数量

    // 5. 循环遍历当前批次的所有粒子
    for (int32 i = 0; i < Context.GetNumInstances(); ++i)
    {
        // 读取输入
        const int32 Index = ParticleIndexHandler.GetAndAdvance(); 

        if (Index >= 0 && Index < NumParticles)
        {
            // 写入输出
            *(OutPositionHandler.GetDestAndAdvance()) = Positions[Index]; 
        }
        else
        {
            // 索引越界
            *(OutPositionHandler.GetDestAndAdvance()) = FVector::ZeroVector; 
        }
    }
}