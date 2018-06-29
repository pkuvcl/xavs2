# xavs2

遵循 `AVS2-P2/IEEE1857.4` 视频编码标准的编码器. 

对应的解码器 **davs2** 可在 [Github][4] 或 [Gitee (mirror in China)][5] 上找到.

[![GitHub tag](https://img.shields.io/github/tag/pkuvcl/xavs2.svg?style=plastic)]()
[![GitHub issues](https://img.shields.io/github/issues/pkuvcl/xavs2.svg)](https://github.com/pkuvcl/xavs2/issues)
[![GitHub forks](https://img.shields.io/github/forks/pkuvcl/xavs2.svg)](https://github.com/pkuvcl/xavs2/network)
[![GitHub stars](https://img.shields.io/github/stars/pkuvcl/xavs2.svg)](https://github.com/pkuvcl/xavs2/stargazers)

## 编译方法
### Windows

可使用`VS2013`打开解决方案`./build/win32/xavs2.sln`进行编译, 也可以使用更新的vs版本打开上述解决方案.
打开解决方案后, 将工程`xavs2`设置为启动项, 进行编译即可. 

#### 注意
1. 首次编译本项目时, 需要安装一个 `shell 执行器`, 比如 `git-for-windows` 中的 `bash`, 
 需要将该 `bash` 所在的目录添加到系统环境变量 `PATH` 中.
 如上所述, 如果您以默认配置安装了`git-for-windows`, 
 那么将 `C:\Program Files\Git\bin` 添加到环境变量中即可.
2. 需要安装 `vsyasm`, 我们建议的版本号是 `1.2.0`, 因为官方更新的版本存在编译问题.
  下载地址: http://yasm.tortall.net/Download.html .
  一个修改过可以正常编译的 `1.3.0` 版本(注意:此修改非官方, 编译请参考yasm的编译指南)可以在这里找到: https://github.com/luofalei/yasm/tree/vs2013 .
  其典型的安装步骤如下(使用VS2013时):
```
(1) 将vsyasm.exe文件拷贝到如下目录: 
    "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\"
(2)	将剩余三个vsyasm文件拷贝到MSBuild模板目录: 
    "C:\Program Files (x86)\MSBuild\Microsoft.Cpp\v4.0\V120\BuildCustomizations\"
(3) 重新打开VS2013, asmopt工程应已正常加载, 编译无错误. 
```

### Linux

对于linux系统, 依次执行如下命令即可完成编译:
```
$ cd build/linux
$ ./configure
$ make
```

## 运行和测试
```
./xavs2 [-f encoder.cfg [-f seq.cfg]] [-p ParameterName=value] [--ParameterName=value]
```

### 使用配置文件进行参数设置
```
./xavs2 -f encoder.cfg -f seq4K.cfg -p InputFile=input.yuv -p FramesToBeEncoded=500 \
  -p preset=0 -p recon=. -p initial_qp=32 -p OutputFile=test.avs
```

### 不使用配置文件
```
./xavs2 -p InputFile=input.yuv --FramesToBeEncoded=500 --FrameRate=6 \
  --SourceWidth=3840 --SourceHeight=2160 --InputSampleBitDepth=8 --SampleBitDepth=8 \
  --thread_frames=1 --thread_rows=1 --preset=0 \
  --recon=. --initial_qp=32 --OutputFile=test.avs
```

## 主页链接

[北京大学-视频编码算法研究室(PKU-VCL)][1]

`AVS2-P2/IEEE1857.4` 编码器: [xavs2 (Github)][2], [xavs2 (mirror in China)][3]

`AVS2-P2/IEEE1857.4` 解码器: [davs2 (Github)][4], [davs2 (mirror in China)][5]

  [1]: http://vcl.idm.pku.edu.cn/ "PKU-VCL"
  [2]: https://github.com/pkuvcl/xavs2 "xavs2 github repository"
  [3]: https://gitee.com/pkuvcl/xavs2 "xavs2 gitee repository"
  [4]: https://github.com/pkuvcl/davs2 "davs2 decoder@github"
  [5]: https://gitee.com/pkuvcl/davs2 "davs2 decoder@gitee"
