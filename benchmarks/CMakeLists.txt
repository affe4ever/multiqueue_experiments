add_library(threading STATIC util/threading.cpp)
target_include_directories(threading PUBLIC "${CMAKE_SOURCE_DIR}/util")
target_link_libraries(threading PUBLIC Threads::Threads)
target_compile_features(threading PUBLIC cxx_std_17)

add_library(benchmark_base INTERFACE)
target_include_directories(benchmark_base INTERFACE "." "..")
target_include_directories(benchmark_base SYSTEM
                           INTERFACE "${CMAKE_SOURCE_DIR}/third_party")
target_compile_definitions(benchmark_base
                           INTERFACE L1_CACHE_LINE_SIZE=${L1_CACHE_LINE_SIZE})
target_compile_definitions(benchmark_base INTERFACE PAGE_SIZE=${PAGE_SIZE})
target_compile_features(benchmark_base INTERFACE cxx_std_17)
target_link_libraries(benchmark_base INTERFACE threading)
if(PAPI_FOUND)
  target_link_libraries(benchmark_base INTERFACE PAPI::PAPI)
  target_compile_definitions(benchmark_base INTERFACE WITH_PAPI)
endif()

add_library(dijkstra_base INTERFACE)
target_sources(dijkstra_base INTERFACE dijkstra.cpp)
target_link_libraries(dijkstra_base INTERFACE benchmark_base)

if(EXP_TUNING_TARGETS)
  add_custom_target(tuning_all)
  add_custom_target(tuning_log_all)
endif()

foreach(benchmark dijkstra)
  add_custom_target(${benchmark}_all)
  add_custom_target(${benchmark}_log_all)
  if(EXP_TUNING_TARGETS)
    add_custom_target(${benchmark}_tuning_all)
    add_custom_target(${benchmark}_log_tuning_all)
  endif()
endforeach()

function(add_benchmark benchmark target)
  add_executable(${benchmark}_${target})
  target_link_libraries(${benchmark}_${target} PRIVATE ${target}
                                                       ${benchmark}_base)

  add_executable(${benchmark}_log_${target})
  target_link_libraries(${benchmark}_log_${target} PRIVATE ${target}
                                                           ${benchmark}_base)
  target_compile_definitions(${benchmark}_log_${target} PRIVATE LOG_OPERATIONS)
endfunction()

foreach(target ${COMPETITORS})
  add_benchmark(dijkstra ${target})
  add_dependencies(dijkstra_all dijkstra_${target})
  add_dependencies(dijkstra_log_all dijkstra_log_${target})
endforeach()

if(EXP_TUNING_TARGETS)
  foreach(target ${TUNING_TARGETS})
    add_benchmark(dijkstra ${target})
    add_dependencies(dijkstra_tuning_all dijkstra_${target})
    add_dependencies(dijkstra_log_tuning_all dijkstra_log_${target})
  endforeach()
endif()

add_executable(dijkstra_seq dijkstra_seq.cpp)
target_link_libraries(dijkstra_seq PRIVATE benchmark_base)
