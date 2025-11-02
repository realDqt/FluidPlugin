// Fill out your copyright notice in the Description page of Project Settings.

using System;
using System.IO;
using UnrealBuildTool;

public class PhysEngineSDK : ModuleRules
{
	public PhysEngineSDK(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
		Type = ModuleType.External;

		// 1. 设置 C++ 17 标准 (等同于 SET(CMAKE_CXX_STANDARD 17))
		CppStandard = CppStandardVersion.Cpp17;
		
		// 2. 添加UE模块依赖
		PublicDependencyModuleNames.AddRange(new string[] { "Core" });
		PrivateDependencyModuleNames.AddRange(new string[] { "CoreUObject", "Engine" });
		
		// (等同于 option() 和 if() 逻辑)
		bool bEnableExtendedAlignedStorage = true; // 手动设置
		if (bEnableExtendedAlignedStorage)
		{
			PublicDefinitions.Add("_ENABLE_EXTENDED_ALIGNED_STORAGE=1");
		}
		else
		{
			PublicDefinitions.Add("_DISABLE_EXTENDED_ALIGNED_STORAGE=1");
		}
		PublicDefinitions.Add("PE_USE_BACKEND_WRAPPER=0");

		// 3. 设置预处理器定义 (等同于 add_definitions)
		PublicDefinitions.Add("GLOBALBENCHMARK=1");
		
		
		// TODO: 这些宏如何设置？
		PublicDefinitions.Add("USE_DOUBLE=0");
		PublicDefinitions.Add("NTDDI_WIN11_DT=1");
		PublicDefinitions.Add("_WIN32_WINNT_WIN10_RS1=1");
		PublicDefinitions.Add("_WIN32_WINNT_WIN10_TH2=2");
		PublicDefinitions.Add("_WIN32_WINNT_WIN10_RS3=3");
		PublicDefinitions.Add("_WIN32_WINNT_WIN10_RS4=4");
		PublicDefinitions.Add("PE_USE_CUDA=5");
		
		

		// 4. 定位你的第三方SDK
		// 这是获取当前.build.cs文件所在目录的标准方法
		string ModulePath = ModuleDirectory;
		
		// 导航到 "Source/ThirdParty/PhysEngineSDK"
		string SdkPath = Path.GetFullPath(Path.Combine(ModulePath));
		
		// --- 5. 设置库和头文件路径 (核心部分) ---
		
		// (等同于 set(ENGINE_LIB_DIR ...))
		string LibPath = Path.Combine(SdkPath, "lib");
		
		// (等同于 set(ENGINE_INCLUDE_DIR ...) 和 set(TASKFLOW_INCLUDE_DIR ...))
		string EngineIncludePath = Path.Combine(SdkPath, "include", "physeng", "src");
		string TaskflowIncludePath = Path.Combine(SdkPath, "include", "physeng", "3rdparty", "taskflow");
		string EnginePath = Path.Combine(SdkPath, "include", "physeng");

		// 添加头文件搜索路径
		PublicIncludePaths.Add(EngineIncludePath);
		PublicIncludePaths.Add(TaskflowIncludePath);
		PublicIncludePaths.Add(EnginePath);

		string MathPath = Path.Combine(EnginePath, "linear_math", "3rdparty", "eigen");
		PublicIncludePaths.Add(MathPath);
		
		MathPath = Path.Combine(EnginePath, "linear_math", "3rdparty", "enoki", "include");
		PublicIncludePaths.Add(MathPath);
		
		// ---------------freeglut begin-------------------
		string FreeglutPath = Path.Combine(EnginePath, "3rdparty", "freeglut");
		string FreeglutSrcPath = Path.Combine(FreeglutPath, "include");
		string FreeglutLibPath = Path.Combine(FreeglutPath, "lib", "x64");
		PublicIncludePaths.Add(FreeglutSrcPath);
		string GlewLibs = Path.Combine(FreeglutLibPath, "glew32.lib");
		PublicAdditionalLibraries.Add(GlewLibs);
		
		string GlutLibs = Path.Combine(FreeglutLibPath, "freeglut.lib"); // debug or release?
		PublicAdditionalLibraries.Add(GlutLibs);
		// ---------------freeglut end-------------------
		

		// 添加库文件搜索路径
		PublicLibraryPaths.Add(LibPath);
		
		// !! 重要 !!
		// 你必须在这里显式列出所有需要链接的 .lib 文件名
		// 我根据你的CMake错误历史推测了几个名字
		PublicAdditionalLibraries.Add(Path.Combine(LibPath, "Release", "EngineCudaCommon.lib"));
		PublicAdditionalLibraries.Add(Path.Combine(LibPath, "Release", "EngineCudaObject.lib"));
		PublicAdditionalLibraries.Add(Path.Combine(LibPath, "Release", "EngineCudaViewer.lib"));
		// ... 在这里添加其他所有来自你SDK的.lib文件
		
		
		/*
		// 6. 添加运行时DLL (等同于 set(ENGINE_BIN_DIR ...))
		// 这些是你的插件在运行时需要加载的.dll文件
		string BinPath = Path.Combine(SdkPath, "include", "physeng", "3rdparty", "freeglut", "bin", "x64");
		
		// 自动查找BinPath下的所有.dll文件
		if (Directory.Exists(BinPath))
		{
			string[] Dlls = Directory.GetFiles(BinPath, "*.dll");
			foreach (string Dll in Dlls)
			{
				// 告诉UE在打包时要包含这个DLL
				RuntimeDependencies.Add(Dll);
				
				string DllName = Path.GetFileName(Dll); // e.g., "glew32.dll"
				string LibName = Path.GetFileNameWithoutExtension(Dll) + ".lib"; // e.g., "glew32.lib"
				string LibFullPath = Path.Combine(FreeglutPath, "lib", "x64", LibName);
				
				if (DllName.Equals("glew32.dll", StringComparison.OrdinalIgnoreCase))
				{
					if (File.Exists(LibFullPath))
					{
						// 静态链接：告诉链接器在启动时就需要这个 .lib
						PublicAdditionalLibraries.Add(LibFullPath);
					}
					else
					{
						System.Console.WriteLine("Warning: 静态链接失败，未找到 .lib 文件: " + LibFullPath);
					}
				}
				else
				{
									
					// 告诉UE在启动时要加载这个DLL
					PublicDelayLoadDLLs.Add(Path.GetFileName(Dll));
				}
			}
		}
		*/

		
		// ---------------CUDA begin-------------------
		// (等同于 enable_language(CUDA) 和 find_package(CUDAToolkit))
		// (等同于 add_definitions(-DPE_USE_CUDA))


		// UE需要知道CUDA SDK在哪里，以便链接cudart.lib等
		string CudaSdkPath = System.Environment.GetEnvironmentVariable("CUDA_PATH");

		if (!string.IsNullOrEmpty(CudaSdkPath))
		{
			PublicIncludePaths.Add(Path.Combine(CudaSdkPath, "include"));
			string CudaCommonPath = Path.Combine(EnginePath, "3rdparty", "cuda", "common");
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
