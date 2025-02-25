# Find doctest package
find_package(doctest REQUIRED)

# Function to create test targets
function(add_einsum_test test_name)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE einsum doctest::doctest)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Add test executables
add_einsum_test(test_parser)
add_einsum_test(test_contraction)
add_einsum_test(test_einsum)

# Add a test that runs all tests
add_executable(test_all
    test_parser.cpp
    test_contraction.cpp
    test_einsum.cpp
)
target_link_libraries(test_all PRIVATE einsum doctest::doctest)
add_test(NAME test_all COMMAND test_all)
