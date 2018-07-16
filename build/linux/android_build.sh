#!/bin/sh
# Instruction:
#   A simple build script of xavs2/davs2 for Android platform.
# Author:
#   Falei LUO <falei.luo@gmail.com>
#
# reference: http://blog.csdn.net/u010963658/article/details/51404710
#            https://github.com/yixia/x264.git
# PIE:       http://stackoverflow.com/questions/30612067/only-position-independent-executables-pie-are-supported
#            https://github.com/danielkop/android-ffmpeg/commit/616a099151fb6be05b559adc4c9ed95afacd92c2

# ------------------------------------------------------
# ARCH configurations: (arm/arm64), sdk-verision (19， 21)
#     only 21 and later version supports arm64
ARCH=arm
SDK_VERSION=19
ANDROID_NDK="/android/ndk-r14b"

# ------------------------------------------------------
if [ "$ARCH" = "arm64" ]
then
    PLATFORM_PREFIX="aarch64-linux-android-"
    HOST="aarch64"
    PLATFORM_VERSION=4.9
    EXTRA_CFLAGS="-march=armv8-a -D__ARM_ARCH_7__ -D__ARM_ARCH_7A__ -fPIE -pie"
else
    PLATFORM_PREFIX="arm-linux-androideabi-"
    HOST="arm"
    PLATFORM_VERSION=4.9
    EXTRA_CFLAGS="-march=armv7-a -mfloat-abi=softfp -mfpu=neon -D__ARM_ARCH_7__ -D__ARM_ARCH_7A__ -fPIE -pie"
fi

PREFIX=$(pwd)/android/${ARCH}

NDKROOT=$ANDROID_NDK/platforms/android-${SDK_VERSION}/arch-${ARCH}
TOOLCHAIN=$ANDROID_NDK/toolchains/${PLATFORM_PREFIX}${PLATFORM_VERSION}/prebuilt/linux-x86_64
CROSS_PREFIX=$TOOLCHAIN/bin/${PLATFORM_PREFIX}
EXTRA_LDFLAGS="-fPIE -pie"

# configure
rm -rf config.mak
./configure --prefix=$PREFIX \
    --cross-prefix=$CROSS_PREFIX \
    --extra-cflags="$EXTRA_CFLAGS" \
    --extra-ldflags="$EXTRA_LDFLAGS" \
    --enable-pic \
    --enable-static \
    --enable-strip \
    --disable-asm \
    --host=arm-linux \
    --sysroot=$NDKROOT

make clean
make STRIP= -j4 # install || exit 1
# scripts ends here

