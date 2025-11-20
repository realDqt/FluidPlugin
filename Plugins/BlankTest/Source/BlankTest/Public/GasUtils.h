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
	vec3r origin;
	Real vorticity;
	Real diffusion;
	Real buoyancy;  //浮力
	Real vcEpsilon;
	Real decreaseDensity;
};

struct GasSystemParams
{
	//GasSource
	vec3r source;
	Real radius;
	vec3r velocity;
	Real density;

	//RenderData
	vec3r color;
	vec3r lightDir;
	Real decay;
	unsigned char ambient;
};
