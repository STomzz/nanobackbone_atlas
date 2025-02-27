#!/bin/bash
ScriptPath="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"
ModelPath="${ScriptPath}/../models"


function build()
{
  if [ -d ${ScriptPath}/../build/intermediates/host ];then
    rm -rf ${ScriptPath}/../build/intermediates/host
  fi

  mkdir -p ${ScriptPath}/../build/intermediates/host
  cd ${ScriptPath}/../build/intermediates/host

  cmake ../../../src -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
  if [ $? -ne 0 ];then
    echo "[ERROR] cmake error, Please check your environment!"
    return 1
  fi
  make
  if [ $? -ne 0 ];then
    echo "[ERROR] build failed, Please check your environment!"
    return 1
  fi
  cd - > /dev/null
}

function main()
{
  echo "[INFO] nanotracker_backbone preparation"

 ret=`find ${ModelPath} -maxdepth 1 -name nanotrack_backbone_om.om 2> /dev/null`

   if [[ ${ret} ]];then
      echo "[INFO] The nanotrack_backbone_om.om already exists.start buiding"
    else
      echo "[ERROR] nanotrack_backbone_om.om does not exist, please follow the readme to convert the model and place it in the correct position!"
      return 1
    fi

  build
  if [ $? -ne 0 ];then
    return 1
  fi

  echo "[INFO] nanotrack_backbone_om.om preparation is complete"
}
main