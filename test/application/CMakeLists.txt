enable_language(CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -lm")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O1 -fsanitize=address -fno-omit-frame-pointer")

add_executable(
  application_test_mlp
  test_mlp.cpp
  ../common/mlp_config.c
)

target_link_libraries(application_test_mlp
  PRIVATE unity
  PRIVATE mlp
)

add_test(NAME application_test_mlp COMMAND application_test_mlp)
set_tests_properties(application_test_mlp PROPERTIES TIMEOUT 3)

