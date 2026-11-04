SCRIPT_FILE=${BASH_SOURCE[0]}
PROJ_DIR=$(cd -- "$( dirname -- "$SCRIPT_FILE" )/.." &> /dev/null && pwd)
BUILD_DIR=${PROJ_DIR}/build

export ASAN_OPTIONS=allocator_may_return_null=1
ctest --test-dir $BUILD_DIR --verbose