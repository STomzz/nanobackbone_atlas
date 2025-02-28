#!/bin/bash
ScriptPath="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"
echo "[INFO] The nanotracker_backbone starts to run"

running_command="gdb ./main"
cd ${ScriptPath}/../out
${running_command}
if [ $? -ne 0 ];then
    echo "[INFO] The program gdb failed"
else
    echo "[INFO] The program gdb successfully"
fi