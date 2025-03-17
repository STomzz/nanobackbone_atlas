import numpy as np

# 读取二进制文件并恢复形状
data1 = np.fromfile('../out/output1.bin', dtype=np.float32).reshape(1, 2, 15, 15)
data2 = np.fromfile('../out/output2.bin', dtype=np.float32).reshape(1, 4, 15, 15)

np.save("../out/output1_qua.npy",data1)
np.save("../out/output2_qua.npy",data2)

# 验证数据
print("输出1形状:", data1.shape)        # 应输出 (1, 2, 15, 15)
print("第一个数据:", data1[0, 0, 0, 0])
print("最后一个数据:", data1[0, 1, 14, 14])

print("输出2形状:", data2.shape)  #应输出 (1, 4, 15, 15)
print("第一个数据:", data2[0, 0, 0, 0])
print("最后一个数据:", data2[0, 3, 14, 14])