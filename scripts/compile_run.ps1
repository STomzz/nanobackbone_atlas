# 定义路径参数
$SourceFile = "./../src/main2.cpp"
$OutputDir = "./../results"
$OutputExe = "$OutputDir/out"

# 检查源码文件是否存在
if (-not (Test-Path $SourceFile -PathType Leaf)) {
    Write-Error "error $SourceFile not found"
    exit 1
}

# 创建输出目录（如果不存在）
if (-not (Test-Path $OutputDir -PathType Container)) {
    New-Item -Path $OutputDir -ItemType Directory -Force | Out-Null
    Write-Host "mkdir outputDir $OutputDir"
}

# 执行编译命令
g++ -o $OutputExe $SourceFile -std=c++11

# 检查编译结果
if ($LASTEXITCODE -ne 0) {
    Write-Error "compile faild"
    exit $LASTEXITCODE
} else {
    Write-Host "compile successfully"
}
# 执行可执行文件
./../results/out

if ($LASTEXITCODE -ne 0) {
    Write-Error "run faild"
    exit $LASTEXITCODE
} else {
    Write-Host "run successfully"
}
