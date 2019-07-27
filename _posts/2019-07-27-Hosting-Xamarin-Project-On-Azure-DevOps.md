---
title: "Automate Xamarin Shared Project to build on Azure DevOps"
date: 2019-07-27 11:30
tags: 
  - Xamarin
  - Azure DevOps
categories: "Xamarin" 
---
This blog is going to talk about how to automate the build process for Xamarin.Forms shared project on Azure DevOps by using yaml file. This blog can be helpful for those who want to publish their shared project as a nuget package without including the native startup project. I will use an example that contains a .netstandard shared project, an iOS library project and an android library project to config a yaml file.

## Automating .Net Standard Project 
```
jobs: 
- job: build_core
  displayName: Build Shared Library
  pool:
    name: 'Hosted Windows 2019 with VS2019'

  steps:
  - task: NuGetToolInstaller@0
    displayName: 'Use NuGet $(NugetVersion)'
    inputs:
      versionSpec: $(NugetVersion)

  - task: NuGetCommand@2
    displayName: 'NuGet restore'
    inputs:
      restoreSolution: $(RestoreSolution)

  - task: VSBuild@1
    displayName: 'Build .net standard Project'
    inputs:
      solution: $(.NetStandardProjectPath)      
      platform: 'AnyCPU'
      configuration: $(BuildConfiguration)

  - task: CopyFiles@2
    inputs:
      SourceFolder: '$(.NetStandardProjectFolderPath)/bin/Release/netstandard2.0/'
      TargetFolder: '$(build.ArtifactStagingDirectory)/lib/netstandard2.0/'

  - task: PublishBuildArtifacts@1
    displayName: 'Publish Artifact'
    inputs:
      artifactName: nuget
      pathToPublish: '$(Build.ArtifactStagingDirectory)'
```
In the yaml file, we need to put ```jobs```at first because we are going to build three projects. Next, we specify a name for this build process and select Windows machine under the pool because VSBuild task can only happen on Windows machine. We first do a nuget restore to resotre all the depencies that current project needs, and then we do a VSBuild. VSBuild will build the project and all the referenced project for you as well. After building the project, we copy the compiled files under bins folder to ArtifactStagingDirectory and publish it. Automating the build process on shared project is done.

## Automating Android Library Project
```
- job: build_android
  displayName: Build Android Library
  pool:
    name: 'Hosted Windows 2019 with VS2019'

  steps:
  - task: NuGetToolInstaller@0
    displayName: 'Use NuGet $(NugetVersion)'
    inputs:
      versionSpec: $(NugetVersion)

  - task: NuGetCommand@2
    displayName: 'NuGet restore'
    inputs:
      restoreSolution: $(RestoreSolution)


  - task: VSBuild@1
    displayName: 'Build Droid Project'
    inputs:
      solution: $(PathToAndroidLibraryProject)      
      msbuildArgs: '/p:JavaSdkDirectory="$(JAVA_HOME_8_X64)"'
      platform: 'AnyCPU'
      configuration: $(BuildConfiguration)

  - task: CopyFiles@2
    inputs:
      SourceFolder: '$(PathToAndroidLibraryProjectFolder)/bin/Release/monoandroid90/90/'
      TargetFolder: '$(build.ArtifactStagingDirectory)/lib/MonoAndroid90/'

  - task: PublishBuildArtifacts@1
    displayName: 'Publish Artifact'
    inputs:
      artifactName: nuget
      pathToPublish: '$(Build.ArtifactStagingDirectory)'
```
Building an android project is quite similar to .net standrad project but there are two things to notice. In VSBuild task, we have to specify the msbuildArgs ```/p:JavaSdkDirectory="$(JAVA_HOME_8_X64)```. Without this arg, Azure pipline will throw you an error say can not find android SDK when building the project. Another thing to notice is when copying files, the compiled files are under ```monoandroid90/90``` which is also a bit different.

## Automating iOS Library Project
```
- job: build_macos
  displayName: Build macOS Library
  pool:
      vmImage: macos-10.13
      
  steps:
      # make sure to select the correct Xamarin and mono
      - bash: sudo $AGENT_HOMEDIRECTORY/scripts/select-xamarin-sdk.sh $(MONO_VERSION)
        displayName: Switch to the latest Xamarin SDK
      - bash: echo '##vso[task.setvariable variable=MD_APPLE_SDK_ROOT;]'/Applications/Xcode_$(XCODE_VERSION).app;sudo xcode-select --switch /Applications/Xcode_$(XCODE_VERSION).app/Contents/Developer
        displayName: Switch to the latest Xcode
      - task: NuGetToolInstaller@0
        displayName: 'Use NuGet $(NugetVersion)'
        inputs:
          versionSpec: $(NugetVersion)

      - task: NuGetCommand@2
        displayName: 'NuGet restore'
        inputs:
          restoreSolution: $(RestoreSolution)

      - task: MSBuild@1
        displayName: 'Build iOS Project'
        inputs:
          solution: $(PathToiOSProject)
          configuration: $(BuildConfiguration)

      - task: CopyFiles@2
        inputs:
          SourceFolder: `$(PathToiOSProjectFolder)/bin/Release/xamarin.ios10/'
          TargetFolder: '$(build.ArtifactStagingDirectory)/lib/xamarin.ios10/'

      - task: PublishBuildArtifacts@1
        displayName: 'Publish Artifact'
        inputs:
          artifactName: nuget
          pathToPublish: '$(Build.ArtifactStagingDirectory)'
```
When building an iOS project, we have to change the machine to macOS under pool otherwise the iOS project will never build.
After changing the host machine, we have to set the correct Xamarin and Mono version for the machine. Notice we are using MSBuild at here because mac hosting machine is not supported to build VSBuild task. Then we do the same thing like previous project build.

## Pack to Nuget Package
After building all the projects, now we need to pack it into nuget package.
```
- job: nuget_pack
  displayName: Nuget Phase

  pool:
    name: 'Hosted Windows 2019 with VS2019'  

  dependsOn:
    - build_core
    - build_android
    - build_macos

  steps:
  - task: NuGetToolInstaller@0
    displayName: 'Use NuGet $(NugetVersion)'
    inputs:
      versionSpec: $(NugetVersion)

  - task: DownloadBuildArtifacts@0
    displayName: 'Download build artifact'
    inputs:
      artifactName: 'nuget'
      downloadPath: '$(Build.ArtifactsDirectory)'

  - task: CopyFiles@2
    displayName: 'Copy Files nuget Library'
    inputs:
      SourceFolder: '$(Build.ArtifactsDirectory)/nuget'
      TargetFolder: 'nuget/'
      OverWrite: true

  - powershell: |
      $buildConfiguration = "Release"
      $nugetVersion = "" + $env:Nuget_Version

      Write-Host("my nuget version is $nugetVersion")
      Write-Host("Update nuspecs")
      Get-ChildItem './nuget/MyNuget.nuspec' -Recurse | Foreach-Object {
            (Get-Content $_) | Foreach-Object  {
                $_ -replace '\$version\$', $nugetVersion `               
          } | Set-Content $_
        }
    env:
      Nuget_Version: $(NugetPackageVersion)
      Commit_Id: $(Build.SourceVersion)

  - task: NuGetCommand@2
    displayName: 'Make NuGet Package'
    inputs:
      command: pack
      packagesToPack: 'nuget/MuNuget.nuspec'
      packDestination: '$(Build.ArtifactStagingDirectory)/nuget/'
      configuration: $(BuildConfiguration)
  - task: NuGetCommand@2
    inputs:
      command: 'push'
      packagesToPush: '$(Build.ArtifactStagingDirectory)/nuget/*.nupkg'
      nuGetFeedType: 'internal'
      publishVstsFeed: $(FeedId)
      
  - task: PublishBuildArtifacts@1
    displayName: 'Publish Artifact: nuget'
    inputs:
      PathtoPublish: '$(Build.ArtifactStagingDirectory)/nuget'
      ArtifactName: nuget
```
First we have to download the artifact that we published in the previous step and copy it to a folder in our repo. And then we run a powerscript to replace the nuget version for our nuget package to the latest build number. Then you can also automate the process to push it to your private Artifacts folder or upload the package to nuget manually.

Thats all you have to do to automate the process of building Xamarin Shared Project and pack it in to nuget package. 
