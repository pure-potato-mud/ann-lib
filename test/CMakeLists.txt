add_executable(unit_test_mlp
  unit/test_mlp.c
)

target_link_libraries(unit_test_mlp
  PRIVATE unity
  PRIVATE mlp
)

target_include_directories(unit_test_mlp
  PRIVATE ${PROJECT_SOURCE_DIR}/src
)

add_test(NAME unit_test_mlp COMMAND unit_test_mlp)
set_tests_properties(unit_test_mlp PROPERTIES TIMEOUT 1)

add_executable(integration_test_mlp
  integration/test_mlp.c
)

target_link_libraries(integration_test_mlp
  PRIVATE unity
  PRIVATE mlp
)

add_test(NAME integration_test_mlp COMMAND integration_test_mlp)
set_tests_properties(integration_test_mlp PROPERTIES TIMEOUT 1)
