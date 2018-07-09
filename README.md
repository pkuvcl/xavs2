# xavs2

**xavs2** is an open-source encoder of `AVS2-P2/IEEE1857.4` video coding standard.

A decoder, **davs2**, can be found at [Github][4] or  [Gitee (mirror in China)][5].

[![GitHub tag](https://img.shields.io/github/tag/pkuvcl/xavs2.svg?style=plastic)]()
[![GitHub issues](https://img.shields.io/github/issues/pkuvcl/xavs2.svg)](https://github.com/pkuvcl/xavs2/issues)
[![GitHub forks](https://img.shields.io/github/forks/pkuvcl/xavs2.svg)](https://github.com/pkuvcl/xavs2/network)
[![GitHub stars](https://img.shields.io/github/stars/pkuvcl/xavs2.svg)](https://github.com/pkuvcl/xavs2/stargazers)

## Build it
### Windows
Use `VS2013` or later version of visual studio to open the solution file `./build/vs2013/xavs2.sln`,
then set the `xavs2` as the start project and build it.

#### Notes
1. A `shell executor`, i.e. the bash in git for windows, is needed and should be found in `PATH` variable.
 For example, the path `C:\Program Files\Git\bin` can be added if git-for-windows is installed.
2. `vsyasm` is needed and `1.2.0` is suggested for windows platform.
 It can be downloaded through: http://yasm.tortall.net/Download.html .
 A later version `1.3.0` (unofficial revision, please read the instructions of `yasm` to build it for your work), can be found in https://github.com/luofalei/yasm/tree/vs2013 .
   The installation of `vsyasm` is as follows (if you were using `VS2013`):
```
(1) Copy `vsyasm.exe` to the following directory, 
    "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\"
(2)	Copy the other 3 files in `vsyasm` to the `MSBuild template` directorty, as follows, 
    "C:\Program Files (x86)\MSBuild\Microsoft.Cpp\v4.0\V120\BuildCustomizations\"
(3) Re-open the solution. 
```

 

### Linux
```
$ cd build/linux
$ ./configure
$ make
```

## Try it
```
./xavs2 [-f encoder.cfg [-f seq.cfg]] [-p ParameterName=value] [--ParameterName=value]
```

### Encode with configuration files
```
./xavs2 -f encoder.cfg -f seq4K.cfg -p InputFile=input.yuv -p FramesToBeEncoded=500 \
  -p preset=0 -p recon=. -p initial_qp=32 -p OutputFile=test.avs
```

### Enocde without configuraton files
```
./xavs2 -p InputFile=input.yuv --FramesToBeEncoded=500 --FrameRate=6 \
  --SourceWidth=3840 --SourceHeight=2160 --InputSampleBitDepth=8 --SampleBitDepth=8 \
  --thread_frames=1 --thread_rows=1 --preset=0 \
  --recon=. --initial_qp=32 --OutputFile=test.avs
```

## Homepages

[PKU-VCL][1]

`AVS2-P2/IEEE1857.4` Encoder: [xavs2 (Github)][2], [xavs2 (mirror in China)][3]

`AVS2-P2/IEEE1857.4` Decoder: [davs2 (Github)][4], [davs2 (mirror in China)][5]

  [1]: http://vcl.idm.pku.edu.cn/ "PKU-VCL"
  [2]: https://github.com/pkuvcl/xavs2 "xavs2 github repository"
  [3]: https://gitee.com/pkuvcl/xavs2 "xavs2 gitee repository"
  [4]: https://github.com/pkuvcl/davs2 "davs2 decoder@github"
  [5]: https://gitee.com/pkuvcl/davs2 "davs2 decoder@gitee"
