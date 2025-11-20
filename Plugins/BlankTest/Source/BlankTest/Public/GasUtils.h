#pragma once

#include "CoreMinimal.h"
#include "object/gas_world.h"

class GasUtils
{
public:
	// UE (厘米/左手系) -> SDK (米/右手系)
	// 假设 SDK 使用 Y 轴向上，UE 使用 Z 轴向上
	static vec3r ToSDK(const FVector& UEVec)
	{
		// 1. 坐标轴重映射: UE(X, Z, Y) -> SDK(X, Y, Z)
		// 2. 单位换算: 厘米 -> 米 (* 0.01f)
		return make_vec3r(UEVec.X * 0.01f, UEVec.Z * 0.01f, UEVec.Y * 0.01f);
	}

	// SDK (米/右手系) -> UE (厘米/左手系)
	static FVector ToUE(const vec3r& SDKVec)
	{
		// 1. 坐标轴重映射: SDK(x, z, y) -> UE(X, Y, Z)
		// 2. 单位换算: 米 -> 厘米 (* 100.0f)
		return FVector(SDKVec.x * 100.0f, SDKVec.z * 100.0f, SDKVec.y * 100.0f);
	}
};

struct GasWorldParams
{
	vec3r origin{make_vec3r(0.0f)};
};

struct GasSystemParams
{
    //BaseParams
	Real vorticity{0.000002f};
	Real diffusion{0.000000f};
	Real buoyancy{4.0f};  //浮力
	Real vcEpsilon{5.0f};
	Real decreaseDensity{0.001f};
	
	//RenderData
	vec3r color{make_vec3r(255 / 255.0f, 255 / 255.0f, 255 / 255.0f)};
	vec3r lightDir{make_float3(0, -1, 0)};
	Real decay{0.06f};
	unsigned char ambient{100};
};
struct GasSourceParams
{
	//GasSource
	vec3r source{make_vec3r(0.0f, -1.0f, 0.0f)};
	Real radius{0.4f};
	vec3r velocity{make_vec3r(0.0f, 0.0f, 0.0f)};
	Real density{1.0f};
};
