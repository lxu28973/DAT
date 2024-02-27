add_executable(gemmChain gemm-chain.cpp)
add_executable(bgemmChain bgemm-chain.cpp)
add_executable(fusedIO fused-io.cpp)
add_executable(gurobi gurobi.cpp)
add_executable(indicator indicator.cpp)
target_link_libraries(gurobi optimized ${GUROBI_CXX_LIBRARY}
        debug ${GUROBI_CXX_DEBUG_LIBRARY})
target_link_libraries(gurobi ${GUROBI_LIBRARY})
target_link_libraries(gurobi ${Boost_LIBRARIES})

add_executable(4DTensor 4d-tensor.cpp)
target_link_libraries(4DTensor optimized ${GUROBI_CXX_LIBRARY}
        debug ${GUROBI_CXX_DEBUG_LIBRARY})
target_link_libraries(4DTensor ${GUROBI_LIBRARY})
target_link_libraries(4DTensor ${Boost_LIBRARIES})

add_executable(fused fused.cpp)
target_link_libraries(fused optimized ${GUROBI_CXX_LIBRARY}
        debug ${GUROBI_CXX_DEBUG_LIBRARY})
target_link_libraries(fused ${GUROBI_LIBRARY})
target_link_libraries(fused ${Boost_LIBRARIES})

add_executable(noFused no-fused.cpp)
target_link_libraries(noFused optimized ${GUROBI_CXX_LIBRARY}
        debug ${GUROBI_CXX_DEBUG_LIBRARY})
target_link_libraries(noFused ${GUROBI_LIBRARY})
target_link_libraries(noFused ${Boost_LIBRARIES})

add_executable(dse dse.cpp)
target_link_libraries(dse optimized ${GUROBI_CXX_LIBRARY}
        debug ${GUROBI_CXX_DEBUG_LIBRARY})
target_link_libraries(dse ${GUROBI_LIBRARY})
target_link_libraries(dse ${Boost_LIBRARIES})

include_directories(../src)

add_test(NAME GemmChain COMMAND gemmChain)
add_test(NAME 4DTensor COMMAND 4DTensor)
add_test(NAME BgemmChain COMMAND bgemmChain)
add_test(NAME FusedIO COMMAND fusedIO)
add_test(NAME Gurobi COMMAND gurobi)
add_test(NAME Fused COMMAND fused)
add_test(NAME NoFused COMMAND noFused)
add_test(NAME DSE COMMAND dse)
