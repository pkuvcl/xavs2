# xavs2

**xavs2** is an open-source `AVS2/IEEE1857.4` video encoder.

[![GitHub tag](https://img.shields.io/github/tag/pkuvcl/xavs2.svg?style=plastic)]()
[![GitHub issues](https://img.shields.io/github/issues/pkuvcl/xavs2.svg)](https://github.com/pkuvcl/xavs2/issues)
[![GitHub forks](https://img.shields.io/github/forks/pkuvcl/xavs2.svg)](https://github.com/pkuvcl/xavs2/network)
[![GitHub stars](https://img.shields.io/github/stars/pkuvcl/xavs2.svg)](https://github.com/pkuvcl/xavs2/stargazers)

## Compiling Method
### Windows
Use VS2013 or latest version of  visual studio open the `./build/vs2013/xavs2.sln` solution and set the `xavs2` as the start project.

#### Notes
1. In the compile process, `shell executor` is needed. We need add it to enviroment variables.
2. `vsyasm` is needed, install it according to your OS version.

### Linux
`Makefile` is a simple way to organize code compilation and it had already exitst in the `./build/linux`. You can perform the following commands.
```
$ cd build/linux
$ ./configure
$ make
```

## Usage
```
./xavs2 [-f encoder.cfg [-f seq.cfg]] [-p ParameterName=value] [--ParameterName=value]
```
### Encode with configure file
```
./xavs2 -f encoder.cfg [-f seq.cfg] -p InputFile=input.yuv -p FramesToBeEncoded=500 -p FrameRate=6 -p SourceWidth=3840 -p SourceHeight=2160 -p InputSampleBitDepth=8 -p SampleBitDepth=8 -p preset=0 -p recon=. -p initial_qp=45 -p OutputFile=test.avs
```
### Enocde without configure file
```
./xavs2 -p InputFile=input.yuv --FramesToBeEncoded=500 --FrameRate=6 --SourceWidth=3840 --SourceHeight=2160 --InputSampleBitDepth=8 --SampleBitDepth=8 --thread_frames=1 --thread_rows=1 --preset=0 --recon=. --initial_qp=45 --OutputFile=test.avs
```
