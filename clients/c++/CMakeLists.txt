cmake_minimum_required(VERSION 3.18)

project(triton-cpp-client)
message(STATUS ${PROJECT_NAME}) 
set(TritonClientThirdParty_DIR "/workspace/build/third-party")
set(TritonClientBuild_DIR "/workspace/install")
set(CMAKE_CXX_STANDARD 17)



set(CURL_DIR "${TritonClientThirdParty_DIR}/curl/lib/cmake/CURL")
set(Protobuf_DIR "${TritonClientThirdParty_DIR}/protobuf/lib/cmake/protobuf")
set(gRPC_DIR "${TritonClientThirdParty_DIR}/grpc/lib/cmake/grpc")
set(c-ares_DIR "${TritonClientThirdParty_DIR}/c-ares/lib/cmake/c-ares")
set(TritonClient_DIR "${TritonClientBuild_DIR}/lib/cmake/TritonClient")
set(TritonCommon_DIR "${TritonClientBuild_DIR}/lib/cmake/TritonCommon")

message(STATUS  ${CURL_DIR})
message(STATUS  ${Protobuf_DIR})
message(STATUS  ${gRPC_DIR})
message(STATUS  ${TritonClient_DIR})

find_package(OpenCV REQUIRED)
find_package(TritonCommon REQUIRED)
find_package(TritonClient REQUIRED)
find_package(gflags REQUIRED)

ADD_LIBRARY(clientSDK SHARED ${PROJECT_SOURCE_DIR}/clientclass.cpp)
target_include_directories(
    clientSDK
    PRIVATE ${OpenCV_INCLUDE_DIRS} ${TritonClientBuild_DIR}/include ${GFLAGS_INCLUDE_DIRS} 
  )
target_link_directories(clientSDK PRIVATE ${TritonClientBuild_DIR}/lib)
target_link_libraries(clientSDK 
PRIVATE
grpcclient
httpclient
${OpenCV_LIBS}
${GFLAGS_LIBRARIES}
)

ADD_LIBRARY(c_clientSDK SHARED ${PROJECT_SOURCE_DIR}/c_wrapper.cpp)
target_include_directories(
  c_clientSDK
    PRIVATE ${OpenCV_INCLUDE_DIRS} ${TritonClientBuild_DIR}/include ${GFLAGS_INCLUDE_DIRS} 
  )
target_link_libraries(c_clientSDK 
PRIVATE
clientSDK
${OpenCV_LIBS}
)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/main.cpp ${PROJECT_SOURCE_DIR}/c_wrapper.cpp ${PROJECT_SOURCE_DIR}/clientclass.cpp)
target_include_directories(
    ${PROJECT_NAME} 
    PRIVATE ${OpenCV_INCLUDE_DIRS} ${TritonClientBuild_DIR}/include ${GFLAGS_INCLUDE_DIRS} 
  )
target_link_directories(${PROJECT_NAME} PRIVATE ${TritonClientBuild_DIR}/lib)
target_link_libraries(${PROJECT_NAME} 
PRIVATE
grpcclient
httpclient
${OpenCV_LIBS}
${GFLAGS_LIBRARIES}
)

