import shutil
import os

# 将输出文件夹打包成 zip
output_archive = '/kaggle/working/features_2000_backup'
shutil.make_archive(output_archive, 'zip', '/kaggle/working/features_2000')

print(f"打包完成：{output_archive}.zip")