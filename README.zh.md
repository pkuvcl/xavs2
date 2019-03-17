# xavs2

遵循 `AVS2-P2/IEEE1857.4` 视频编码标准的编码器. 

对应的解码器 **davs2** 可在 [Github][4] 或 [Gitee (mirror in China)][5] 上找到.

[![GitHub tag](https://img.shields.io/github/tag/pkuvcl/xavs2.svg?style=plastic)]()
[![GitHub issues](https://img.shields.io/github/issues/pkuvcl/xavs2.svg)](https://github.com/pkuvcl/xavs2/issues)
[![GitHub forks](https://img.shields.io/github/forks/pkuvcl/xavs2.svg)](https://github.com/pkuvcl/xavs2/network)
[![GitHub stars](https://img.shields.io/github/stars/pkuvcl/xavs2.svg)](https://github.com/pkuvcl/xavs2/stargazers)
[![Travis Build Status](https://travis-ci.org/pkuvcl/xavs2.svg?branch=master)](https://travis-ci.org/pkuvcl/xavs2)
[![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/qysemawbynvhiktf?svg=true)](https://ci.appveyor.com/project/luofalei/xavs2/build/artifacts)

## 编译方法
### Windows

可使用`VS2013`打开解决方案`./build/win32/xavs2.sln`进行编译, 也可以使用更新的vs版本打开上述解决方案.
打开解决方案后, 将工程`xavs2`设置为启动项, 进行编译即可. 

#### 注意
1. 首次编译本项目时, 需要安装一个 `shell 执行器`, 比如 `git-for-windows` 中的 `bash`, 
 需要将该 `bash` 所在的目录添加到系统环境变量 `PATH` 中.
 如上所述, 如果您以默认配置安装了`git-for-windows`, 
 那么将 `C:\Program Files\Git\bin` 添加到环境变量中即可.
2. 需将 `nasm.exe`放入到 `build/vs2013`目录, `nasm`版本号需为`2.13`或更新.
  对于`windows`平台,可下载如下压缩包中，解压得到`nasm.exe`.
https://www.nasm.us/pub/nasm/releasebuilds/2.14.02/win64/nasm-2.14.02-win64.zip

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
./xavs2 -p InputFile=input.yuv --FramesToBeEncoded=500 --fps=50 \
  --SourceWidth=3840 --SourceHeight=2160 --InputSampleBitDepth=8 --SampleBitDepth=8 \
  --thread_frames=1 --thread_rows=1 --preset=0 \
  --recon=. --initial_qp=32 --OutputFile=test.avs
```

## Issue & Pull Request

欢迎提交 issue，请写清楚遇到问题的环境与运行参数，包括操作系统环境、编译器环境等，重现的流程，
如果可能提供原始输入YUV/码流文件，请尽量提供以方便更快地重现结果。

[反馈问题的 issue 请按照模板格式填写][6]。

如果有开发能力，建议在本地调试出错的代码，并[提供相应修正的 Pull Request][7]。

### 已知问题与工作清单

- 高比特精度(10-bit)支持与其SIMD指令优化.

- 码率控制.

- 场景切换检测与自适应帧类型选择.

- ARM平台的NEON指令优化.

- 等等.

## 主页链接

[北京大学-视频编码算法研究室(PKU-VCL)][1]

`AVS2-P2/IEEE1857.4` 编码器: [xavs2 (Github)][2], [xavs2 (mirror in China)][3]

`AVS2-P2/IEEE1857.4` 解码器: [davs2 (Github)][4], [davs2 (mirror in China)][5]

  [1]: http://vcl.idm.pku.edu.cn/ "PKU-VCL"
  [2]: https://github.com/pkuvcl/xavs2 "xavs2 github repository"
  [3]: https://gitee.com/pkuvcl/xavs2 "xavs2 gitee repository"
  [4]: https://github.com/pkuvcl/davs2 "davs2 decoder@github"
  [5]: https://gitee.com/pkuvcl/davs2 "davs2 decoder@gitee"
  [6]: https://github.com/pkuvcl/xavs2/issues "report issues"
  [7]: https://github.com/pkuvcl/xavs2/pulls "pull request"
