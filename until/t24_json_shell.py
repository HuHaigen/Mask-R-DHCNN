import json
import os


# 1.原有的文件不能改，只能在另一个文件夹生成新的文件（问题不大） 在另一个文件夹中创建好相应的文件路径
# 2.文件的路径都是静态的，需要动态获取的 （重点解决）
#     a 获取到文件夹中有多少个json文件，拿到之后外层做个循环

def process_json(input_json_file, output_json_file):
    file_in = open(input_json_file, "r")
    file_out = open(output_json_file, "w")
    # load数据到变量json_data
    json_data = json.load(file_in)
    shapes = json_data["shapes"]
    num = len(shapes)
    print(num)
    for i in range(num):
        print("old name", shapes[i]["label"])
        shapes[i]["label"] = "cell" + str(i)
        print("new name", shapes[i]["label"])
    file_out.write(json.dumps(json_data))
    file_in.close()
    file_out.close()


root_val_path = "F://python//t24_label//train//json//"
file_names = os.listdir(root_val_path)
print(file_names)
for name in file_names:
    path_1 = root_val_path+name
    path_2 = "F://python//t24_m//train//json//"+name
    process_json(path_1, path_2)

