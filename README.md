# xavs2

**xavs2** is an open-source encoder of `AVS2-P2/IEEE1857.4` video coding standard.

A decoder, **davs2**, can be found at [Github][4] or  [Gitee (mirror in China)][5].

[![GitHub tag](https://img.shields.io/github/tag/pkuvcl/xavs2.svg?style=plastic)]()
[![GitHub issues](https://img.shields.io/github/issues/pkuvcl/xavs2.svg)](https://github.com/pkuvcl/xavs2/issues)
[![GitHub forks](https://img.shields.io/github/forks/pkuvcl/xavs2.svg)](https://github.com/pkuvcl/xavs2/network)
[![GitHub stars](https://img.shields.io/github/stars/pkuvcl/xavs2.svg)](https://github.com/pkuvcl/xavs2/stargazers)
[![Travis Build Status](https://travis-ci.org/pkuvcl/xavs2.svg?branch=master)](https://travis-ci.org/pkuvcl/xavs2)
[![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/qysemawbynvhiktf?svg=true)](https://ci.appveyor.com/project/luofalei/xavs2/build/artifacts)

## Build it
### Windows
Use `VS2013` or later version of visual studio to open the solution file `./build/vs2013/xavs2.sln`,
then set the `xavs2` as the start project and build it.

#### Notes
1. A `shell executor`, i.e. the bash in git for windows, is needed and should be found in `PATH` variable.
 For example, the path `C:\Program Files\Git\bin` can be added if git-for-windows is installed.
2. `nasm.exe` with version `2.13` (or later version) is needed and should be put into the `build/vs2013` directory.
 For windows platform, you can downloaded the packege and unpack the zip file to get `nasm.exe`:
https://www.nasm.us/pub/nasm/releasebuilds/2.14.02/win64/nasm-2.14.02-win64.zip


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
./xavs2 -p InputFile=input.yuv --FramesToBeEncoded=500 --fps=50 \
  --SourceWidth=3840 --SourceHeight=2160 --InputSampleBitDepth=8 --SampleBitDepth=8 \
  --thread_frames=1 --thread_rows=1 --preset=0 \
  --recon=. --initial_qp=32 --OutputFile=test.avs
```

## How to Report Bugs and Provide Feedback

Use the ["Issues" tab on Github][6].

## How to Contribute

We welcome community contributions to this project. Thank you for your time! By contributing to the project, you agree to the license and copyright terms therein and to the release of your contribution under these terms.

If you have some bugs or features fixed, and would like to share with the public, please [make a Pull Request][7].

### Contribution process

-  Validate that your changes do not break a build

-  Perform smoke tests and ensure they pass

-  Submit a pull request for review to the maintainer

### Known workitems or bugs

- high bit-depth (i.e. 10-bit) support and SIMD optimization.

- Rate-control in CBR, VBR.

- Adaptive scene change detection and frame type decision.

- NEON support for ARM platform.

- and so on.

## Homepages

[PKU-VCL][1]

`AVS2-P2/IEEE1857.4` Encoder: [xavs2 (Github)][2], [xavs2 (mirror in China)][3]

`AVS2-P2/IEEE1857.4` Decoder: [davs2 (Github)][4], [davs2 (mirror in China)][5]

  [1]: http://vcl.idm.pku.edu.cn/ "PKU-VCL"
  [2]: https://github.com/pkuvcl/xavs2 "xavs2 github repository"
  [3]: https://gitee.com/pkuvcl/xavs2 "xavs2 gitee repository"
  [4]: https://github.com/pkuvcl/davs2 "davs2 decoder@github"
  [5]: https://gitee.com/pkuvcl/davs2 "davs2 decoder@gitee"
  [6]: https://github.com/pkuvcl/xavs2/issues "report issues"
  [7]: https://github.com/pkuvcl/xavs2/pulls "pull request"
