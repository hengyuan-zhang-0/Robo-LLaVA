import pyarrow.parquet as pq
import glob
from tqdm import tqdm
import os
# 读取 parquet 文件
def read_parquet(file_path):
    # 使用 pyarrow 打开 parquet 文件
    table = pq.read_table(file_path)
    return table.to_pandas()


def save_image_and_text(savepath, index, image_bytes, question, answer, image_filename="output.jpg", question_filename="output.txt", answer_filename="output.txt"):
    # 保存 image 字节流为 JPG 文件
    with open(os.path.join(savepath,'images',image_filename), "wb") as image_file:
        image_file.write(image_bytes)

    # 将 question 和 answer 写入文本文件
    with open(os.path.join(savepath, question_filename), "a+") as text_file:
        text_file.write(f"{question}\n")
    
    with open(os.path.join(savepath, answer_filename), "a+") as text_file:
        text_file.write(f"{answer}\n")

# 遍历每条数据并输出key和value的类型
def iterate_data(dataframe, filename):
    saveroot = '/share/pd/Dataset/EgoThink/parsed_test'
    
    savepath = os.path.join(saveroot, filename.split('_test_')[0])
    os.makedirs(savepath, exist_ok=True)
    os.makedirs(os.path.join(savepath,'images'), exist_ok=True)
    for index, row in tqdm(dataframe.iterrows()):
        # if index>0:
        #     break
        # print(f"Row {index}:")
        # for key in dataframe.columns:
        # print(type(row['image']), row['image']['path'])
        save_image_and_text(savepath, index, row['image']['bytes'], row['question'], row['answer'],image_filename=f"{index}.jpg", question_filename="question.txt", answer_filename="answer.txt")
            # value = row[key]
            # print(f"Key: {key}, Value: {value}, Type: {type(value)}")
        # print("-" * 50)  # 分隔符

if __name__ == "__main__":
    # 替换为你的parquet文件路径
    root_path = "/share/pd/Dataset/EgoThink/test/"
    file_paths = glob.glob(os.path.join(root_path, "*.parquet"))
    for file_path in file_paths:
        dataframe = read_parquet(file_path)
        filename = file_path.split('/')[-1].split('.')[0]
        iterate_data(dataframe, filename)
