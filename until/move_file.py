import os, shutil


# 获取文件夹下所有文件名称，返回list
def get_names(obj_path):
    files = os.listdir(obj_path)
    names = []
    for file_ in files:
        names.append(str(file_.split(".")[0]))
    names.sort()
    print(names)
    # os.close(10)
    return names


# 移动文件到新的文件夹
def move_file(path_, obj_path):
    base_path = u"C://Users//admin//Desktop//论文//数据集//t24_label//t24_label"
    json_path = base_path + "//json//"
    mask_path = base_path + "//label//"
    json_file_path = base_path + "//label_unzip//"
    if base_path and base_path != "":  # 判断文件路径是否为空
        # 获取需要移动的文件名称
        names = get_names(path_)
        file_list = len(names)
        # 循环移动
        for name in names:
            # 生成文件名称
            json_name = "Frame_" + name + ".json"
            mask_name = "label" + name + ".png"
            json_file_name = "Frame_" + name + "_json"

            # 移动json文件
            shutil.move(json_path + json_name, obj_path)  # 移动文件
            # 移动mask文件
            shutil.move(mask_path + mask_name, obj_path)  # 移动文件
            # 移动json_file文件
            shutil.move(json_file_path + json_file_name, obj_path)  # 移动文件
        print('分组完成')  # 弹出提示框
    else:
        print('请选择文件夹')  # 弹出提示框


val_path = "H://t24//val"
train_path = "H://t24//train"
val_obj_path = u"C://Users//admin//Desktop//论文//数据集//t24_label//val"
train_obj_path = u"C://Users//admin//Desktop//论文//数据集//t24_label//train"
move_file(val_path, val_obj_path)
move_file(train_path, train_obj_path)
