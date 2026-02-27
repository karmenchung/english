import csv

# 读取对话文本文件
with open('dialogs.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 准备 CSV 文件
with open('dialogs.csv', 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)

    # 写入 CSV 文件的标题行
    csvwriter.writerow(['english', 'chinese'])

    for line in lines:
        if line.strip():  # 跳过空行
            parts = line.strip().rsplit(' ', 1)  # 从右边按最后一个空格进行分割
            if len(parts) == 2:
                english, chinese = parts
                csvwriter.writerow([english.strip(), chinese.strip()])

print("CSV 文件已生成: .csv")



