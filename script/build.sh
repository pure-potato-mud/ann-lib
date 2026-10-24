SCRIPT_FILE=${BASH_SOURCE[0]}
SOURCE_FILE=$0
PROJ_DIR=$(cd -- "$( dirname -- "$SCRIPT_FILE" )/.." &> /dev/null && pwd)
BUILD_DIR=${PROJ_DIR}/build

leave() {
  local status=$1
  if [ $SCRIPT_FILE = $SOURCE_FILE ]; then
    exit $status
  else
    return $status
  fi
}

echo_result() {
  local status=$1
  local success_text=$2
  local fail_text=$3
  if [ $status -eq 0 ]; then
    if [[ $success_text != "" ]]; then
      echo $success_text
    fi
  else
    if [[ $fail_text != "" ]]; then
      echo $fail_text
    fi
  fi
  return $status
}

# configure
if [ "$1" == "--debug" ]; then
  cmake -S $PROJ_DIR -B $BUILD_DIR
else
  cmake -S $PROJ_DIR -B $BUILD_DIR -DCMAKE_BUILD_TYPE=Debug
fi
configure_status=$?
echo_result $configure_status "" "configure failed."
if [[ $configure_status -ne 0 ]]; then
echo hi
  return $(leave)
fi

# build
cmake --build $BUILD_DIR
build_status=$?
echo_result $build_status "" "build failed."
if [[ $build_status -ne 0 ]]; then
  return $(leave)
fi

# run(optional)
if [ "$1" = "--run" ]; then
  source $PROJ_DIR/script/run.sh
fi
