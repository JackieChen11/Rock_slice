# Rock_slice
岩石薄片  目前只有分类  分割算法还没添加
和RESnet101 模型相关的是train.py   
predict.py   
model.py  (给出3种模型定义)
spilt_files.py （将每一个数据集都分为train和val  比例9：1  最好只运行一次 免得重复分割）
move_files.py  (将所有后缀为train的数据集移至train_folder  带val的移至val_folder)
predict.py(接收一张图片  给出属于class:0x  x取值0~9d对应10种岩石)
class_indices.json  储存10种岩石的分类结果  
fine_tune.py 根据predict的运行结果  与真实结果进行毕较 储存反馈  
