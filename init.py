# 数据预处理
!mkdir work/datas
!mkdir work/train
!mkdir work/datas/Butterfly20_test
!mkdir work/datas/Butterfly20
#解压文件夹
!unzip -qo data/data126285/Butterfly20_test.zip -d work/datas/Butterfly20_test
!unzip -qo data/data126285/Butterfly20.zip -d work/datas/Butterfly20

import os
import re
train_path = 'work/datas/Butterfly20/Butterfly20'
genus_path = 'work/datas/Butterfly20/genus.txt'
spicies_path = 'work/datas/Butterfly20/species.txt'
target_path = 'work/datas/Butterfly20/data_list.txt'
#建立样本数据读取路径与样本标签之间的关系
species_dict={}
genus_dict={}
#配置spices_dict
with open(spicies_path) as f:
    for line in f:
        a,b = line.strip("\n").split(" ")
        species_dict[b]=a
print('species_dict:')
print(species_dict)
print()
#配置genus_dict
with open(genus_path) as f:
    for line in f:
        a,b = line.strip("\n").split(" ")
        genus_dict[b]=a
print('genus_dict:')
print(genus_dict)
print()
#配置data_list
data_list = []#数据格式[[path,genus_index,species_index][...]...]
class_list = species_dict.keys()
print('class_list:')
print(class_list)
print()
for each in class_list:
    genus_key = each.split('.')[1].split('_')[0]
    for f in os.listdir(train_path+'/'+each):
        data_list.append([train_path+'/'+each+'/'+f,genus_dict[genus_key],species_dict[each]]) 
print('data_list')
print(data_list)

#将data_list写入target_path的txt中
with open(target_path,'w') as f:
    for line in data_list:
        f.write(line[0]+' '+str(line[1])+' '+str(line[2])+'\n')
#