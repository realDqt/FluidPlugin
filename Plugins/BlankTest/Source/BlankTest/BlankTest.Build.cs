// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;
using System.IO;

public class BlankTest : ModuleRules
{
	public BlankTest(ReadOnlyTargetRules Target) : base(Target)
	{
		// 默认设置
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
		CppStandard = CppStandardVersion.Cpp17;
		
		
		// TODO: 这些宏如何设置？
		PublicDefinitions.Add("NTDDI_WIN11_DT=1");
		PublicDefinitions.Add("_WIN32_WINNT_WIN10_RS1=1");
		PublicDefinitions.Add("_WIN32_WINNT_WIN10_TH2=2");
		PublicDefinitions.Add("_WIN32_WINNT_WIN10_RS3=3");
		PublicDefinitions.Add("_WIN32_WINNT_WIN10_RS4=4");
		PublicDefinitions.Add("PE_USE_CUDA=5");

		string PhysEnginePath = Path.Combine(ModuleDirectory, "Private", "PhysEngine");
		string EngineIncludePath = Path.Combine(PhysEnginePath, "src");
		string TaskflowIncludePath = Path.Combine(PhysEnginePath, "3rdparty", "taskflow");
		
		PublicIncludePaths.Add(EngineIncludePath);
		PublicIncludePaths.Add(TaskflowIncludePath);
		
		string MathPath = Path.Combine(PhysEnginePath, "linear_math", "3rdparty", "eigen");
		PublicIncludePaths.Add(MathPath);
		
		MathPath = Path.Combine(PhysEnginePath, "linear_math", "3rdparty", "eigen", "test");
		PublicIncludePaths.Add(MathPath);
		
		MathPath = Path.Combine(PhysEnginePath, "linear_math", "3rdparty", "enoki", "include");
		PublicIncludePaths.Add(MathPath);
		
		PublicIncludePaths.AddRange(
			new string[] {
				// ... add public include paths required here ...
			}
			);
				
		
		PrivateIncludePaths.AddRange(
			new string[] {
				// ... add other private include paths required here ...
			}
			);
			
		
		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				// ... add other public dependencies that you statically link with here ...
			}
			);
			
		
		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"CoreUObject",
				"Engine",
				"Slate",
				"SlateCore",
				// ... add private dependencies that you statically link with here ...	
			}
			);
		
		
		DynamicallyLoadedModuleNames.AddRange(
			new string[]
			{
				// ... add any modules that your module loads dynamically here ...
			}
			);
		
		
		// ---------------CUDA begin-------------------
		// (等同于 enable_language(CUDA) 和 find_package(CUDAToolkit))
		// (等同于 add_definitions(-DPE_USE_CUDA))


		// UE需要知道CUDA SDK在哪里，以便链接cudart.lib等
		string CudaSdkPath = System.Environment.GetEnvironmentVariable("CUDA_PATH");

		if (!string.IsNullOrEmpty(CudaSdkPath))
		{
			PublicIncludePaths.Add(Path.Combine(CudaSdkPath, "include"));
			string CudaCommonPath = Path.Combine(PhysEnginePath, "3rdparty", "cuda", "common");
			string CudaIncPath = Path.Combine(CudaCommonPath, "inc");
			PublicIncludePaths.Add(CudaIncPath);

			string CudaCommonLibPath = Path.Combine(CudaCommonPath, "lib", "x64");
			PublicAdditionalLibraries.Add(Path.Combine(CudaCommonLibPath, "freeglut.lib"));
			PublicAdditionalLibraries.Add(Path.Combine(CudaCommonLibPath, "glew64.lib"));
			
			// 1. 定义 CUDA lib 路径
			string CudaLibPath = Path.Combine(CudaSdkPath, "lib", "x64");
			PublicLibraryPaths.Add(CudaLibPath);
			
			// 链接CUDA运行时库 (你的PhysEngine库依赖它)
			PublicAdditionalLibraries.Add("cudart_static.lib");
			// 你可能还需要: "cublas.lib", "cufft.lib" 等, 
			// 取决于你的PhysEngine到底用了什么
			
			

			// 2. 删除 PublicLibraryPaths.Add(...) 这一行

			// 3. 在 PublicAdditionalLibraries 中提供完整路径
			PublicAdditionalLibraries.Add(Path.Combine(CudaLibPath, "cudart_static.lib"));
    
			// 你可能还需要: 
			PublicAdditionalLibraries.Add(Path.Combine(CudaLibPath, "cublas.lib"));
			PublicAdditionalLibraries.Add(Path.Combine(CudaLibPath, "cufft.lib"));
			PublicAdditionalLibraries.Add(Path.Combine(CudaLibPath, "cuda.lib"));
		}
		else
		{
			// 如果找不到CUDA，构建会失败，并给出清晰的提示
			throw new BuildException("PhysEngine plugin requires the NVIDIA CUDA SDK. Please install it and set the CUDA_PATH environment variable.");
		}
		// ---------------CUDA end-------------------
	}
}
