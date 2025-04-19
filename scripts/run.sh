#!/bin/bash
# 作用：获取当前脚本所在的目录路径，并将其存储在变量   ScriptPath   中。
ScriptPath="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"

echo "[INFO] Nanotrack starts to run"
running_command="./main ../data/tv_tuanliu.mkv ../results/output_tv_tuanliu.mp4 706 679 155 141"
# running_command="./main ../data/girl_dance.mp4 ../results/output_girl_dance.mp4 275 149 62 60"
cd ${ScriptPath}/../bin
${running_command}
if [ $? -ne 0 ];then
    echo "[INFO] The program runs failed"
else
    echo "[INFO] The program runs successfully"
fi