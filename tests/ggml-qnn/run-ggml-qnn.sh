#!/bin/bash

#modify following lines to adapt to local dev envs
QNN_SDK_PATH=/opt/qcom/aistack/qnn/2.20.0.240223/

GGML_QNN_TEST=ggml-qnn-test
REMOTE_PATH=/data/local/tmp/

adb push ${GGML_QNN_TEST} ${REMOTE_PATH}
adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnSystem.so ${REMOTE_PATH}/
adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnCpu.so ${REMOTE_PATH}/
adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnGpu.so ${REMOTE_PATH}/

#the QNN HTP(aka DSP) backend only verified on Xiaomi14(Qualcomm SM8650-AB Snapdragon 8 Gen 3) successfully
adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtp.so ${REMOTE_PATH}/
adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtpNetRunExtensions.so ${REMOTE_PATH}/
adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtpPrepare.so ${REMOTE_PATH}/
adb push ${QNN_SDK_PATH}/lib/aarch64-android/libQnnHtpV75Stub.so ${REMOTE_PATH}/
adb push ${QNN_SDK_PATH}/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so ${REMOTE_PATH}/

adb shell chmod +x /data/local/tmp/${GGML_QNN_TEST}
adb shell /data/local/tmp/${GGML_QNN_TEST}
