# Makefile

include config.mak

vpath %.c $(SRCPATH)
vpath %.h $(SRCPATH)
vpath %.S $(SRCPATH)
vpath %.asm $(SRCPATH)
vpath %.rc $(SRCPATH)
CFLAGS += -I. -I$(SRCPATH) \
		-I$(SRCPATH)/common \
		-I$(SRCPATH)/common/x86 \
		-I$(SRCPATH)/common/vec \
		-I$(SRCPATH)/encoder \
		-I$(SRCPATH)/test

GENERATED =

all: default
default:

# common sources
SRCS = \
	common/block_info.c common/common.c \
	common/cpu.c common/cudata.c \
	common/cg_scan.c \
	common/frame.c common/intra.c common/mc.c \
	common/pixel.c common/quant.c \
	common/threadpool.c common/transform.c \
	common/win32thread.c \
	common/primitives.c \
	common/filter_alf.c \
	common/filter_deblock.c \
	common/filter_sao.c

# encoder sources
SRCS += \
	encoder/aec_ctx.c encoder/aec.c \
	encoder/aec_rdo.c encoder/aec_fastrdo.c encoder/aec_vrdo.c \
	encoder/alf.c \
	encoder/encoder.c \
	encoder/encoder_report.c \
    encoder/header.c \
	encoder/me.c encoder/ratecontrol.c \
	encoder/sao.c encoder/wquant.c \
	encoder/md_intra.c \
	encoder/md_inter.c \
	encoder/presets.c \
	encoder/pre_encode.c \
	encoder/rdo.c encoder/rdoq.c encoder/rps.c \
	encoder/slice.c \
	encoder/tdrdo.c \
	encoder/yuv_writer.c \
	encoder/wrapper.c \
	encoder/xavs2.c encoder/xavs2_api.c \
	encoder/xlist.c \
	encoder/parameters.c

SRCCLI = test/test.c

SRCSO =
OBJS =
OBJAVX =
OBJSO =
OBJCLI =

#OBJCHK = tools/checkasm.o

CONFIG: $(shell cat config.h)

ifneq ($(findstring HAVE_THREAD 1, $(CONFIG)),)
SRCS    += common/threadpool.c
endif

# MMX/SSE optims
ifneq ($(AS),)
# asm --------------------------------------------------------------
X86SRC = common/x86/blockcopy8.asm \
		common/x86/const-a.asm \
		common/x86/cpu-a.asm \
		common/x86/dct8.asm \
		common/x86/mc-a.asm \
		common/x86/mc-a2.asm \
		common/x86/pixel-32.asm \
		common/x86/pixel-a.asm \
		common/x86/pixel-util8.asm \
		common/x86/pixeladd8.asm \
		common/x86/quant8.asm \
		common/x86/sad-a.asm \
		common/x86/sad-vpp.asm \
		common/x86/satd-a.asm \
		common/x86/ssd-a.asm

ifeq ($(SYS_ARCH),X86)
ARCH_X86 = yes
ASMSRC   = $(X86SRC) 
endif

## Until now, we do not have 64-bit asm
ifeq ($(SYS_ARCH),X86_64)
ARCH_X86 = yes
SRCS += common/vec/intrinsic.c \
		common/vec/intrinsic_alf.c \
		common/vec/intrinsic_sao.c \
		common/vec/intrinsic_deblock.c \
		common/vec/intrinsic_inter_pred.c \
		common/vec/intrinsic_intra-pred.c \
		common/vec/intrinsic_intra-filledge.c \
		common/vec/intrinsic_idct.c \
		common/vec/intrinsic_dct.c \
		common/vec/intrinsic_quant.c \
		common/vec/intrinsic_cg_scan.c \
		common/vec/intrinsic_mad.c \
		common/vec/intrinsic_pixel.c

SRCSAVX = common/vec/intrinsic_dct_avx.c \
		common/vec/intrinsic_idct_avx2.c \
		common/vec/intrinsic_quant_avx2.c \
		common/vec/intrinsic_pixel_avx.c \
		common/vec/intrinsic_cg_scan_avx.c \
		common/vec/intrinsic_deblock_avx2.c \
		common/vec/intrinsic_sao_avx2.c \
		common/vec/intrinsic_inter_pred_avx2.c \
		common/vec/intrinsic_intra-pred_avx2.c

CFLAGS += -mmmx -msse -msse2 -msse3 -mssse3 -msse4 -msse4.1 -msse4.2 -msse4a
# ASMSRC   = $(X86SRC:-32.asm=-64.asm)
ASMSRC   = $(X86SRC)
ASFLAGS += -DARCH_X86_64=1
OBJASM  = $(ASMSRC:%.asm=%.o)
$(OBJASM): common/x86/x86inc.asm common/x86/x86util.asm
endif

ifdef ARCH_X86
ASFLAGS += -I$(SRCPATH)/common/x86/
#SRCS    += common/x86/mc-c.c common/x86/predict-c.c
OBJASM  = $(ASMSRC:%.asm=%.o)
$(OBJASM): common/x86/x86inc.asm common/x86/x86util.asm
endif

# AltiVec optims
ifeq ($(SYS_ARCH),PPC)
SRCS += common/ppc/mc.c common/ppc/pixel.c common/ppc/dct.c \
        common/ppc/quant.c common/ppc/deblock.c \
        common/ppc/predict.c
endif

# NEON optims
ifeq ($(SYS_ARCH),ARM)
# x264 ARM asm sources
# ASMSRC += common/arm/cpu-a.S common/arm/pixel-a.S common/arm/mc-a.S \
#           common/arm/dct-a.S common/arm/quant-a.S common/arm/deblock-a.S \
#           common/arm/predict-a.S common/arm/bitstream-a.S
# SRCS   += common/arm/mc-c.c common/arm/predict-c.c
# x265 ARM asm sources
ASMSRC += common/arm/blockcopy8.S common/arm/cpu-a.S common/arm/dct-a.S \
          common/arm/ipfilter8.S common/arm/mc-a.S common/arm/pixel-util.S \
          common/arm/sad-a.S common/arm/ssd-a.S
OBJASM  = $(ASMSRC:%.S=%.o)
endif

# AArch64 NEON optims
ifeq ($(SYS_ARCH),AARCH64)
ASMSRC += common/aarch64/bitstream-a.S \
          common/aarch64/cabac-a.S     \
          common/aarch64/dct-a.S     \
          common/aarch64/deblock-a.S \
          common/aarch64/mc-a.S      \
          common/aarch64/pixel-a.S   \
          common/aarch64/predict-a.S \
          common/aarch64/quant-a.S
SRCS   += common/aarch64/asm-offsets.c \
          common/aarch64/mc-c.c        \
          common/aarch64/predict-c.c
OBJASM  = $(ASMSRC:%.S=%.o)
OBJCHK += tools/checkasm-aarch64.o
endif

# MSA optims
ifeq ($(SYS_ARCH),MIPS)
ifneq ($(findstring HAVE_MSA 1, $(CONFIG)),)
SRCS += common/mips/mc-c.c common/mips/dct-c.c \
        common/mips/deblock-c.c common/mips/pixel-c.c \
        common/mips/predict-c.c common/mips/quant-c.c
endif
endif

# asm --------------------------------------------------------------
endif 
# here ends ifneq ($(AS),)

ifneq ($(HAVE_GETOPT_LONG),1)
SRCS += compat/getopt/getopt.c
endif

## Windows Dll
## ifeq ($(SYS), WINDOWS)
## # OBJCLI += $(if $(RC), xavs2res.o)
## ifneq ($(SONAME),)
## SRCSO   += xavs2dll.c
## OBJSO   += $(if $(RC), xavs2res.dll.o)
## endif
## endif

OBJS   += $(SRCS:%.c=%.o)
OBJAVX += $(SRCSAVX:%.c=%.o)
OBJCLI += $(SRCCLI:%.c=%.o)
OBJSO  += $(SRCSO:%.c=%.o)

.PHONY: all default fprofiled clean distclean install install-* uninstall cli lib-* etags

cli: xavs2$(EXE)
lib-static: $(LIBXAVS2)
lib-shared: $(SONAME)

$(LIBXAVS2): $(GENERATED) .depend $(OBJS) $(OBJAVX) $(OBJASM)
	@echo "\033[33m [linking static] $(LIBXAVS2) \033[0m"
	rm -f $(LIBXAVS2)
	$(AR)$@ $(OBJS) $(OBJAVX) $(OBJASM)
	$(if $(RANLIB), $(RANLIB) $@)

$(SONAME): $(GENERATED) .depend $(OBJS) $(OBJAVX) $(OBJASM) $(OBJSO)
	@echo "\033[33m [linking shared] $(SONAME) \033[0m"
	$(LD)$@ $(OBJS) $(OBJAVX) $(OBJASM) $(OBJSO) $(SOFLAGS) $(LDFLAGS)

ifneq ($(EXE),)
.PHONY: xavs2 checkasm
xavs2: xavs2$(EXE)
checkasm: checkasm$(EXE)
endif

xavs2$(EXE): $(GENERATED) .depend $(OBJCLI) $(CLI_LIBXAVS2)
	@echo "\033[33m [linking execution] xavs2$(EXE) \033[0m"
	$(LD)$@ $(OBJCLI) $(CLI_LIBXAVS2) $(LDFLAGSCLI) $(LDFLAGS)

checkasm$(EXE): $(GENERATED) .depend $(OBJCHK) $(LIBXAVS2)
	@echo "\033[33m [linking checkasm] checkasm$(EXE) \033[0m"
	$(LD)$@ $(OBJCHK) $(LIBXAVS2) $(LDFLAGS)

$(OBJS) $(OBJAVX) $(OBJASM) $(OBJSO) $(OBJCLI) $(OBJCHK): .depend

%.o: %.asm common/x86/x86inc.asm common/x86/x86util.asm
	@echo "\033[33m [Compiling asm]: $< \033[0m"
	$(AS) $(ASFLAGS) -o $@ $<
	-@ $(if $(STRIP), $(STRIP) -x $@) # delete local/anonymous symbols, so they don't show up in oprofile

%.o: %.S
	@echo "\033[33m [Compiling asm]: $< \033[0m"
	$(AS) $(ASFLAGS) -o $@ $<
	-@ $(if $(STRIP), $(STRIP) -x $@) # delete local/anonymous symbols, so they don't show up in oprofile

%.dll.o: %.rc xavs2.h
	@echo "\033[33m [Compiling dll]: $< \033[0m"
	$(RC) $(RCFLAGS)$@ -DDLL $<

%.o: %.rc xavs2.h
	@echo "\033[33m [Compiling rc]: $< \033[0m"
	$(RC) $(RCFLAGS)$@ $<

$(OBJAVX):
	@echo "\033[33m [Compiling]: $(@:.o=.c) \033[0m"
	$(CC) $(CFLAGS) -mavx -mavx2 -c -o $@ $(SRCPATH)/$(@:.o=.c)
	
%.o: %.c
	@echo "\033[33m [Compiling]: $< \033[0m"
	$(CC) $(CFLAGS) -c -o $@ $<

.depend: config.mak
	@rm -f .depend
	@echo "\033[33m dependency file generation... \033[0m"
ifeq ($(COMPILER),CL)
	@$(foreach SRC, $(addprefix $(SRCPATH)/, $(SRCS) $(SRCCLI) $(SRCSO)), $(SRCPATH)/tools/msvsdepend.sh "$(CC)" "$(CFLAGS)" "$(SRC)" "$(SRC:$(SRCPATH)/%.c=%.o)" 1>> .depend;)
	@$(foreach SRC, $(addprefix $(SRCPATH)/, $(SRCSAVX)), $(CC) $(CFLAGS) -mavx2 $(SRC) $(DEPMT) $(SRC:$(SRCPATH)/%.c=%.o) $(DEPMM) 1>> .depend;)
else
	@$(foreach SRC, $(addprefix $(SRCPATH)/, $(SRCS) $(SRCCLI) $(SRCSO)), $(CC) $(CFLAGS) $(SRC) $(DEPMT) $(SRC:$(SRCPATH)/%.c=%.o) $(DEPMM) 1>> .depend;)
	@$(foreach SRC, $(addprefix $(SRCPATH)/, $(SRCSAVX)), $(CC) $(CFLAGS) -mavx2 $(SRC) $(DEPMT) $(SRC:$(SRCPATH)/%.c=%.o) $(DEPMM) 1>> .depend;)
endif

config.mak:
	./configure

depend: .depend
ifneq ($(wildcard .depend),)
include .depend
endif

SRC2 = $(SRCS) $(SRCCLI)
# These should cover most of the important codepaths
OPT0 = --crf 30 -b1 -m1 -r1 --me dia --no-cabac --direct temporal --ssim --no-weightb
OPT1 = --crf 16 -b2 -m3 -r3 --me hex --no-8x8dct --direct spatial --no-dct-decimate -t0  --slice-max-mbs 50
OPT2 = --crf 26 -b4 -m5 -r2 --me hex --cqm jvt --nr 100 --psnr --no-mixed-refs --b-adapt 2 --slice-max-size 1500
OPT3 = --crf 18 -b3 -m9 -r5 --me umh -t1 -A all --b-pyramid normal --direct auto --no-fast-pskip --no-mbtree
OPT4 = --crf 22 -b3 -m7 -r4 --me esa -t2 -A all --psy-rd 1.0:1.0 --slices 4
OPT5 = --frames 50 --crf 24 -b3 -m10 -r3 --me tesa -t2
OPT6 = --frames 50 -q0 -m9 -r2 --me hex -Aall
OPT7 = --frames 50 -q0 -m2 -r1 --me hex --no-cabac

ifeq (,$(VIDS))
fprofiled:
	@echo 'usage: make fprofiled VIDS="infile1 infile2 ..."'
	@echo 'where infiles are anything that xavs2 understands,'
	@echo 'i.e. YUV with resolution in the filename, y4m, or avisynth.'
else
fprofiled:
	$(MAKE) clean
	$(MAKE) xavs2$(EXE) CFLAGS="$(CFLAGS) $(PROF_GEN_CC)" LDFLAGS="$(LDFLAGS) $(PROF_GEN_LD)"
	$(foreach V, $(VIDS), $(foreach I, 0 1 2 3 4 5 6 7, ./xavs2$(EXE) $(OPT$I) --threads 1 $(V) -o $(DEVNULL) ;))
ifeq ($(COMPILER),CL)
# Because Visual Studio timestamps the object files within the PGD, it fails to build if they change - only the executable should be deleted
	rm -f xavs2$(EXE)
else
	rm -f $(SRC2:%.c=%.o)
endif
	$(MAKE) CFLAGS="$(CFLAGS) $(PROF_USE_CC)" LDFLAGS="$(LDFLAGS) $(PROF_USE_LD)"
	rm -f $(SRC2:%.c=%.gcda) $(SRC2:%.c=%.gcno) *.dyn pgopti.dpi pgopti.dpi.lock *.pgd *.pgc
endif

clean:
	rm -f $(OBJS) $(OBJASM) $(OBJCLI) $(OBJSO) $(SONAME) 
	rm -f *.a *.lib *.exp *.pdb libxavs2.so* xavs2 xavs2.exe .depend TAGS
	rm -f checkasm checkasm.exe $(OBJCHK) $(GENERATED) xavs2_lookahead.clbin
	rm -f example example.exe $(OBJEXAMPLE)
	rm -f $(SRC2:%.c=%.gcda) $(SRC2:%.c=%.gcno) *.dyn pgopti.dpi pgopti.dpi.lock *.pgd *.pgc

distclean: clean
	rm -f config.mak xavs2_config.h config.h config.log xavs2.pc xavs2.def conftest*

install-cli: cli
	$(INSTALL) -d $(DESTDIR)$(bindir)
	$(INSTALL) xavs2$(EXE) $(DESTDIR)$(bindir)

install-lib-dev:
	$(INSTALL) -d $(DESTDIR)$(includedir)
	$(INSTALL) -d $(DESTDIR)$(libdir)
	$(INSTALL) -d $(DESTDIR)$(libdir)/pkgconfig
	$(INSTALL) -m 644 $(SRCPATH)/xavs2.h $(DESTDIR)$(includedir)
	$(INSTALL) -m 644 xavs2_config.h $(DESTDIR)$(includedir)
	$(INSTALL) -m 644 xavs2.pc $(DESTDIR)$(libdir)/pkgconfig

install-lib-static: lib-static install-lib-dev
	$(INSTALL) -m 644 $(LIBXAVS2) $(DESTDIR)$(libdir)
	$(if $(RANLIB), $(RANLIB) $(DESTDIR)$(libdir)/$(LIBXAVS2))

install-lib-shared: lib-shared install-lib-dev
ifneq ($(IMPLIBNAME),)
	$(INSTALL) -d $(DESTDIR)$(bindir)
	$(INSTALL) -m 755 $(SONAME) $(DESTDIR)$(bindir)
	$(INSTALL) -m 644 $(IMPLIBNAME) $(DESTDIR)$(libdir)
else ifneq ($(SONAME),)
	ln -f -s $(SONAME) $(DESTDIR)$(libdir)/libxavs2.$(SOSUFFIX)
	$(INSTALL) -m 755 $(SONAME) $(DESTDIR)$(libdir)
endif

uninstall:
	rm -f $(DESTDIR)$(includedir)/xavs2.h $(DESTDIR)$(includedir)/xavs2_config.h $(DESTDIR)$(libdir)/libxavs2.a
	rm -f $(DESTDIR)$(bindir)/xavs2$(EXE) $(DESTDIR)$(libdir)/pkgconfig/xavs2.pc
ifneq ($(IMPLIBNAME),)
	rm -f $(DESTDIR)$(bindir)/$(SONAME) $(DESTDIR)$(libdir)/$(IMPLIBNAME)
else ifneq ($(SONAME),)
	rm -f $(DESTDIR)$(libdir)/$(SONAME) $(DESTDIR)$(libdir)/libxavs2.$(SOSUFFIX)
endif

etags: TAGS

TAGS:
	etags $(SRCS)
