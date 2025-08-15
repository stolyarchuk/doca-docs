#!/bin/bash

#
# Copyright (c) 2022-2024 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

set -e -x

# This script uses the dpacc tool (located in /opt/mellanox/doca/tools/dpacc) to compile DPA kernels device code (for DPA samples).
# This script takes the following arguments:
# arg1: Absolute paths of PCC device source code directory (our code)
# arg2: The project's build path (for the PCC Device build)
# arg3: DOCA Libraries directories path
# arg4: Name of compiled DPA program
# arg5: Absolute paths of directory to keep host stubs
# arg6: Flag to indicate enabling TX counter sampling
# arg7: Flag to indicate enabling updating CC rate from notification point RX bytes
# arg8: DPACC MCPU flag

####################
## Configurations ##
####################

PCC_APP_DEVICE_SRC_DIR=$1
APPLICATION_DEVICE_BUILD_DIR=$2
DOCA_LIB_DIR=$3
PCC_APP_NAME=$4
PCC_DEV_STUBS_KEEP_DIR=$5
ENABLE_TX_COUNTER_SAMPLING=$6
ENABLE_NP_RX_RATE=$7
DPACC_MCPU_FLAG=$8

# Tools location - DPACC, DPA compiler
MLNX_INSTALL_PATH="/opt/mellanox"
DOCA_INSTALL_DIR="${MLNX_INSTALL_PATH}/doca"
DOCA_TOOLS="${DOCA_INSTALL_DIR}/tools"
DPACC="${DOCA_TOOLS}/dpacc"
FLEXIO_INCLUDE="${MLNX_INSTALL_PATH}/flexio/include"

# DOCA include list
DOCA_APP_DEVICE_COMMON_DIR="${PCC_APP_DEVICE_SRC_DIR}/../../common/device"
DOCA_APP_DEVICE_PCC_DIR="${PCC_APP_DEVICE_SRC_DIR}"
DOCA_PCC_DEVICE_RP_ALGO_DIR="${DOCA_APP_DEVICE_PCC_DIR}/rp/algo"
DOCA_INC_LIST="-I${DOCA_INSTALL_DIR}/include/ -I${DOCA_APP_DEVICE_PCC_DIR} -I${DOCA_APP_DEVICE_COMMON_DIR}"

# Set source files
if [ ${PCC_APP_NAME} = "pcc_rp_rtt_template_app" ]
then
        DOCA_PCC_DEV_LIB_NAME="doca_pcc_dev"
        PCC_DEV_RP_RTT_TEMPLATE_DIR=${PCC_APP_DEVICE_SRC_DIR}/rp/rtt_template
        PCC_APP_DEVICE_SRCS=`ls ${PCC_DEV_RP_RTT_TEMPLATE_DIR}/*.c`
        PCC_APP_DEVICE_ALGO_SRCS=`ls ${PCC_DEV_RP_RTT_TEMPLATE_DIR}/algo/*.c`
        PCC_DEVICE_SRC_FILES="${PCC_APP_DEVICE_SRCS} ${PCC_APP_DEVICE_ALGO_SRCS}"
        APP_INC_LIST="${DOCA_INC_LIST} -I${PCC_DEV_RP_RTT_TEMPLATE_DIR}/algo"
        if [ ${AMALGAMATION_BUILD_MODE} = "true" ]; then
                APP_INC_LIST="${APP_INC_LIST} -I${DOCA_PCC_DIR}/device/include/rp -I${DOCA_PCC_DIR}/device/adb_gen/"
        fi
elif [ ${PCC_APP_NAME} = "pcc_rp_switch_telemetry_app" ]
then
        DOCA_PCC_DEV_LIB_NAME="doca_pcc_dev"
        PCC_DEV_RP_SWITCH_TELEM_DIR=${PCC_APP_DEVICE_SRC_DIR}/rp/switch_telemetry
        PCC_APP_DEVICE_SRCS=`ls ${PCC_DEV_RP_SWITCH_TELEM_DIR}/*.c`
        PCC_APP_DEVICE_ALGO_SRCS=`ls ${PCC_DEV_RP_SWITCH_TELEM_DIR}/algo/*.c`
        PCC_DEVICE_SRC_FILES="${PCC_APP_DEVICE_SRCS} ${PCC_APP_DEVICE_ALGO_SRCS}"
        APP_INC_LIST="${DOCA_INC_LIST} -I${PCC_DEV_RP_SWITCH_TELEM_DIR}/algo"
        if [ ${AMALGAMATION_BUILD_MODE} = "true" ]; then
                APP_INC_LIST="${APP_INC_LIST} -I${DOCA_PCC_DIR}/device/include/rp -I${DOCA_PCC_DIR}/device/adb_gen/"
        fi
elif [ ${PCC_APP_NAME} = "pcc_np_nic_telemetry_app" ]
then
        DOCA_PCC_DEV_LIB_NAME="doca_pcc_np_dev"
        PCC_DEV_NP_NIC_TELEM_DIR=${PCC_APP_DEVICE_SRC_DIR}/np/nic_telemetry
        PCC_APP_DEVICE_SRCS=${PCC_DEV_NP_NIC_TELEM_DIR}/np_nic_telemetry_dev_main.c
        PCC_DEVICE_SRC_FILES="${PCC_APP_DEVICE_SRCS}"
        APP_INC_LIST="${DOCA_INC_LIST}"
        if [ ${AMALGAMATION_BUILD_MODE} = "true" ]; then
                APP_INC_LIST="${APP_INC_LIST} -I${DOCA_PCC_DIR}/device/include/np"
        fi
elif [ ${PCC_APP_NAME} = "pcc_np_switch_telemetry_app" ]
then
        DOCA_PCC_DEV_LIB_NAME="doca_pcc_np_dev"
        PCC_DEV_NP_SWITCH_TELEM_DIR=${PCC_APP_DEVICE_SRC_DIR}/np/switch_telemetry
        PCC_APP_DEVICE_SRCS=${PCC_DEV_NP_SWITCH_TELEM_DIR}/np_switch_telemetry_dev_main.c
        PCC_DEVICE_SRC_FILES="${PCC_APP_DEVICE_SRCS}"
        APP_INC_LIST="${DOCA_INC_LIST}"
        if [ ${AMALGAMATION_BUILD_MODE} = "true" ]; then
                APP_INC_LIST="${APP_INC_LIST} -I${DOCA_PCC_DIR}/device/include/np"
        fi
fi

# DPA Configurations
HOST_CC_FLAGS="-Wno-deprecated-declarations -Werror -Wall -Wextra -DFLEXIO_ALLOW_EXPERIMENTAL_API"
DEV_CC_EXTRA_FLAGS="-DSIMX_BUILD,-ffreestanding,-mcmodel=medany,-ggdb,-O2,-DE_MODE_LE,-Wdouble-promotion"
DEVICE_CC_FLAGS="-Wno-deprecated-declarations -Werror -Wall -Wextra -DFLEXIO_DEV_ALLOW_EXPERIMENTAL_API ${DEV_CC_EXTRA_FLAGS}"
DEVICE_SOURCES_STUB_FLAGS="-Wno-attributes -Wno-pedantic -Wno-unused-parameter -Wno-return-type -fPIC"
DEVICE_EXECS_STUB_FLAGS="-Wno-attributes -Wno-pedantic -Wno-implicit-function-declaration -fPIC -nostdlib"

# App flags

DOCA_PCC_SAMPLE_TX_BYTES=""
if [ ${ENABLE_TX_COUNTER_SAMPLING} = "true" ]
then
	DOCA_PCC_SAMPLE_TX_BYTES="-DDOCA_PCC_SAMPLE_TX_BYTES"
fi

DOCA_PCC_NP_RX_RATE=""
if [ ${ENABLE_NP_RX_RATE} = "true" ]
then
	DOCA_PCC_NP_RX_RATE="-DDOCA_PCC_NP_RX_RATE"
fi

APP_FLAGS="${DOCA_PCC_SAMPLE_TX_BYTES}, ${DOCA_PCC_NP_RX_RATE}"

function generate_prog_from_stubs()
{
	COMPILED_SRCS=$1
        DEV_STUB_OBJ_FILES=""
        APP_HOST_STUBS_OUT="${PCC_APP_NAME}_host_stubs"

        cd $PCC_DEV_STUBS_KEEP_DIR

	# extract host stub objects from source stub files
	for F in $COMPILED_SRCS
	do
		F_NAME=`basename $F`
		F_NAME="${F_NAME%.*}"
                F_OBJ_OUT="${F_NAME}.dpa.o"
		gcc -c ${F_NAME}.dpa.host.c -I ${PCC_DEV_STUBS_KEEP_DIR} -D__DPA_OBJ_STUB_FILE__="\"${F_NAME}.stub.inc\"" \
		-o ${F_OBJ_OUT} -I ${FLEXIO_INCLUDE} ${DEVICE_SOURCES_STUB_FLAGS} \
                && objcopy --remove-section=.dpa_obj ${F_OBJ_OUT}
                DEV_STUB_OBJ_FILES+="${F_NAME}.dpa.o "
	done

	# generate application host stub object from app stub files and meta
	gcc -r ${PCC_APP_NAME}.meta.c ${DEV_STUB_OBJ_FILES} -I ${PCC_DEV_STUBS_KEEP_DIR} \
        -D__DPA_EXEC_STUB_FILE__="\"device_exec.stub.inc\"" -o ${APP_HOST_STUBS_OUT}.o -I ${FLEXIO_INCLUDE} ${DEVICE_EXECS_STUB_FLAGS}
        
        ar cr ${PCC_APP_NAME}.a ${APP_HOST_STUBS_OUT}.o

        cd -
}

##################
## Script Start ##
##################

mkdir -p $APPLICATION_DEVICE_BUILD_DIR

# Compile the DPA (kernel) device source code using the DPACC
$DPACC \
-flto \
$PCC_DEVICE_SRC_FILES \
-o ${APPLICATION_DEVICE_BUILD_DIR}/${PCC_APP_NAME}.a \
-mcpu=${DPACC_MCPU_FLAG} \
-hostcc=gcc \
-hostcc-options="${HOST_CC_FLAGS}" \
--devicecc-options="${DEVICE_CC_FLAGS}, ${APP_FLAGS}, ${APP_INC_LIST}" \
-disable-asm-checks \
-device-libs="-L${DOCA_LIB_DIR} -l${DOCA_PCC_DEV_LIB_NAME}" \
--app-name="${PCC_APP_NAME}" \
--keep-dir="${PCC_DEV_STUBS_KEEP_DIR}"

# generate device application program from auto-generated host stubs
generate_prog_from_stubs "${PCC_DEVICE_SRC_FILES}"
