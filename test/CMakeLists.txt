add_executable(test_mlp
  test_mlp.c
)

target_link_libraries(test_mlp
  PRIVATE unity
  PRIVATE mlp
)

target_include_directories(test_mlp
  PRIVATE ${PROJECT_SOURCE_DIR}/src
)

set_tests_properties(PROPERTIES CTEST_TEST_TIMEOUT 1)
add_test(NAME test_mlp COMMAND test_mlp)
