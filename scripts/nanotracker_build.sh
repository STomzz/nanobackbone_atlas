#!/bin/bash
ScriptPath="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"
ModelPath="${ScriptPath}/../models"
ModelFile="nanotrack_deploy_model_nchw.om"
BuildPath="${ScriptPath}/../build"


function build()
{
  if [ -d "${ScriptPath}/../out/" ];then
  rm -rf "${ScriptPath}/../out/"*
  fi
  
  if [ -d ${BuildPath} ];then
    rm -rf ${BuildPath}
  fi

  mkdir -p ${BuildPath}
  cd ${BuildPath}

  cmake ../src -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE

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

 ret=`find ${ModelPath} -maxdepth 1 -name "${ModelFile}" 2> /dev/null`

   if [[ ${ret} ]];then
      echo "[INFO] The "${ModelFile}" already exists.start buiding"
    else
      echo "[ERROR] "${ModelFile}" does not exist, please follow the readme to convert the model and place it in the correct position!"
      return 1
    fi

  build
  if [ $? -ne 0 ];then
    return 1
  fi

  echo "[INFO] "${ModelFile}" preparation is complete"
}
main