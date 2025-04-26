import os
if __name__=="__main__":
    original_directory = os.getcwd()
    print(f"当前目录：{original_directory}")

    # 切换到目标目录
    targetpath="resources/UnifiedToolHub"
    os.chdir(targetpath)
    print(f"切换至：{targetpath}")
    # 执行命令
    os.system(f'''python run.py evaluate test.py''')

    # 切换回原来的目录
    os.chdir(original_directory)
    print(f"切换回 ：{original_directory}")