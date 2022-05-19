
find_package(LevelZero REQUIRED)
find_package(Boost COMPONENTS log REQUIRED)

FILE(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
FILE(GLOB SYCL_KERNELS "${CMAKE_CURRENT_SOURCE_DIR}/sycl_kernels/*.cpp")
add_executable(test_lz_module ${SOURCES} ${SYCL_KERNELS})
target_compile_options(test_lz_module PUBLIC "-fsycl")
target_compile_options(test_lz_module PUBLIC "-O3")
target_compile_options(test_lz_module PUBLIC "-DBOOST_LOG_DYN_LINK")

target_link_libraries(test_lz_module PUBLIC "sycl")
target_link_options(test_lz_module PUBLIC "-fsycl")
target_link_options(test_lz_module PUBLIC "-pthread")


target_link_libraries(test_lz_module PRIVATE ze_loader)
target_link_libraries(test_lz_module PRIVATE Boost::log)

#function(add_lzt_test)
#    set(oneValueArgs NAME GROUP EXTENDED)
#    set(multiValueArgs SOURCES INCLUDE_DIRECTORIES LINK_LIBRARIES KERNELS MEDIA)
#    cmake_parse_arguments(ADD_LZT_TEST
#            "${options}" "${oneValueArgs}" "${multiValueArgs}"
#            ${ARGN}
#            )
#
#    assert_valid_test_group_name("${ADD_LZT_TEST_GROUP}")
#
#    if(DEFINED EXTENDED_BUILD)
#        if(NOT DEFINED ADD_LZT_TEST_EXTENDED)
#            message(STATUS "Ignoring ${ADD_LZT_TEST_NAME} because it is not an extended test")
#            return()
#        endif()
#    endif()
#
#    is_test_group_member("${GROUP}" "${ADD_LZT_TEST_GROUP}" in_scope)
#    if(NOT in_scope)
#        message(STATUS "Ignoring ${ADD_LZT_TEST_NAME} because it is not in ${GROUP}")
#        return()
#    else()
#        message(STATUS "Adding ${ADD_LZT_TEST_NAME}")
#    endif()
#
#
#    add_lzt_test_executable(
#            NAME ${ADD_LZT_TEST_NAME}
#            GROUP ${ADD_LZT_TEST_GROUP}
#            SOURCES ${ADD_LZT_TEST_SOURCES}
#            INCLUDE_DIRECTORIES ${ADD_LZT_TEST_INCLUDE_DIRECTORIES}
#            LINK_LIBRARIES ${ADD_LZT_TEST_LINK_LIBRARIES}
#            KERNELS ${ADD_LZT_TEST_KERNELS}
#            MEDIA ${ADD_LZT_TEST_MEDIA}
#            EXTENDED  ${ADD_LZT_TEST_EXTENDED}
#    )
#
#    file(APPEND
#            "${MANIFEST_BUILD_PATH}"
#            "${ADD_LZT_TEST_NAME}${CMAKE_NEWLINE}"
#            )
#endfunction()
#add_lzt_test(
#  NAME ze_bandwidth
#  GROUP "/perf_tests"
#  SOURCES
#    ../common/src/ze_app.cpp
#    src/ze_bandwidth.cpp
#    src/options.cpp
#  LINK_LIBRARIES ${OS_SPECIFIC_LIBS}
#)
