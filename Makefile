BUILD_DIR ?= build
CMAKE ?= cmake
CMAKE_BUILD_TYPE ?= Release
CMAKE_ARGS ?=
BUILD_ARGS ?=

.PHONY: all configure build debug test install clean

all: build

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) $(CMAKE_ARGS)

build: configure
	$(CMAKE) --build $(BUILD_DIR) --parallel $(BUILD_ARGS)

debug:
	$(MAKE) BUILD_DIR=build/debug CMAKE_BUILD_TYPE=Debug build

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

install: build
	$(CMAKE) --install $(BUILD_DIR)

clean:
	$(CMAKE) -E remove_directory $(BUILD_DIR)
