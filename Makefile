# Strata 定制线构建。分支锚定上游 release tag（见 slot-protect-v0.2.2）。
# 常用：make build | make test | make deploy | make all | make clean
#
# 工具链在仓库 .venv 里（cmake/ninja），nvcc 在 /usr/local/cuda-13.3（不在 PATH）。
# 产物两个：strata（主引擎）与 strata-vision（图片编码器，serve 配置 vision.exe 引用）。

CUDA_DIR      ?= /usr/local/cuda-13.3
NVCC          ?= $(CUDA_DIR)/bin/nvcc
CUDA_ARCH     ?= 89
GGML_DIR      ?= $(CURDIR)/third_party/llama.cpp
BUILD_DIR     ?= build-cuda
VISION_DIR    ?= build-vision
ENGINE_DIR    ?= engine

# 与 setup.py 的源码编译对齐：STRATA_Q6K_EXPERTS/STRATA_MMQ_KQUANTS 必开，产物才等价。
CMAKE_DEFS := -DCMAKE_BUILD_TYPE=Release \
              -DSTRATA_ENABLE_CUDA=ON \
              -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH) \
              -DCMAKE_CUDA_COMPILER=$(NVCC) \
              -DSTRATA_GGML_DIR=$(GGML_DIR) \
              -DSTRATA_BUILD_TESTS=OFF \
              -DSTRATA_Q6K_EXPERTS=ON \
              -DSTRATA_MMQ_KQUANTS=ON

# 与 setup.py 的 vision 编译对齐（setup.py:5456 区）。
VISION_DEFS := -DCMAKE_BUILD_TYPE=Release \
               -DLLAMA_DIR=$(GGML_DIR) \
               -DSTRATA_VISION_CUDA=ON \
               -DSTRATA_PORTABLE=OFF \
               -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH) \
               -DCMAKE_CUDA_COMPILER=$(NVCC)

export PATH := $(CURDIR)/.venv/bin:$(PATH)

.PHONY: all configure build-vision build test deploy deploy-vision clean clean-configure check-engine

all: build deploy

configure:
	cmake -S . -B $(BUILD_DIR) -G Ninja $(CMAKE_DEFS)

build-vision:
	cmake -S tools/vision -B $(VISION_DIR) -G Ninja $(VISION_DEFS)
	ninja -C $(VISION_DIR) strata-vision

build: configure build-vision
	ninja -C $(BUILD_DIR) strata
	@$(MAKE) --no-print-directory check-engine

# GPU 上的真实验证：设计验收 + serve 层单测（有 serve 在跑时注意内存）。
test: build
	$(CURDIR)/.venv/bin/python tools/test_slot_protect.py
	$(CURDIR)/.venv/bin/python -m unittest serve.test_slot_alloc serve.test_parallel \
	    serve.test_slots serve.test_parallel_yield serve.test_images

# 把两个引擎放进部署位。serve 正在跑时不要做（旧进程持有旧文件，新配置重启后才生效）。
deploy: build
	install -D -m 755 $(BUILD_DIR)/strata $(ENGINE_DIR)/strata
	install -D -m 755 $(VISION_DIR)/bin/strata-vision $(ENGINE_DIR)/strata-vision
	@echo "deployed -> $(ENGINE_DIR)/{strata,strata-vision}  (重启 serve 生效)"

# 只补 vision（主引擎未变时）。
deploy-vision: build-vision
	install -D -m 755 $(VISION_DIR)/bin/strata-vision $(ENGINE_DIR)/strata-vision
	@echo "deployed -> $(ENGINE_DIR)/strata-vision"

check-engine:
	@lived=$$($(BUILD_DIR)/strata --version 2>/dev/null | head -1); \
	[ -n "$$lived" ] || { echo "ERROR: $(BUILD_DIR)/strata 运行失败"; exit 1; }; \
	echo "engine: $$lived"

clean:
	rm -rf $(BUILD_DIR) $(VISION_DIR)

# configure 级翻车（换编译器/选项后 cache 冲突）时的重置：只清配置缓存。
clean-configure:
	rm -rf $(BUILD_DIR)/CMakeCache.txt $(BUILD_DIR)/CMakeFiles \
	       $(VISION_DIR)/CMakeCache.txt $(VISION_DIR)/CMakeFiles
