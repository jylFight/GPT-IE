import random
import copy
import torch
from collections import Counter
import openai
from zss import simple_distance, Node
openai.api_base=''
openai.api_key = ''    #这个用closeai的接口，不需要翻墙


def similiar(gold_sent:str,sent_data:list,model,tokenizer):                      #输出应该是tensor       
    gold_input = torch.tensor(tokenizer(gold_sent)['input_ids']).unsqueeze(0)    #直接1,word_num,768
    with torch.no_grad():
        bert_output=model(input_ids=gold_input)[0]
    gold_tensor = torch.mean(bert_output,dim=1)                                  #1,768
    #这里bert得到嵌入的过程会比较慢，但得到cos-sim还挺快的
    sent_result=torch.stack(sent_data).squeeze(1)
    cos_sim = torch.nn.functional.cosine_similarity(gold_tensor,sent_result,dim=1)    
    #这个函数要注意一下，gold-tensor的第一个维度要么为1要么和sent_result维度一样，最后得到的结果是sent_result的初始维度
    return cos_sim
#测试通过

#获得信息，这个也算是共用的，根据example可以得到不同的演示
def info(train_data,example):
    result=''
    for sent in example:
        for data in train_data:
            train_sent=' '.join(data['tokens'])   #句子
            if sent==train_sent:     
                if len(data['relations'])==0:
                    temp_result='User:'+sent+' You are expected to output: <>; '
                else:
                    head_num = data['relations'][0]['head']
                    tail_num = data['relations'][0]['tail']
                    head_enti = ' '.join(data['tokens'][data['entities'][head_num]['start']:data['entities'][head_num]['end']])
                    tail_enti = ' '.join(data['tokens'][data['entities'][tail_num]['start']:data['entities'][tail_num]['end']])
                    rel = data['relations'][0]['type']
                    temp_result='User:'+sent+' You are expected to output: <'+head_enti+','+rel+','+tail_enti+'>; '
                    #这里的格式还真的得想一下，不过算完成
                    #演示里概念其实可以加上类型，可以多一个信息来利用，毕竟概念类型对提取有帮助
                result=result+temp_result
    return result    
    
#先把测试集的结果记录一下，这个也算公共的，放在这儿吧
def get_test_features(test_data):
    features=[]
    for data in test_data:
        sent=' '.join(data['tokens'])
        if len(data['relations'])==0:
            relations='<>'
        else:
            relations=''
            #print(len(data['relations']))
            for i in data['relations']:
                head_num = i['head']
                tail_num = i['tail']
                head_enti = ' '.join(data['tokens'][data['entities'][head_num]['start']:data['entities'][head_num]['end']])
                tail_enti = ' '.join(data['tokens'][data['entities'][tail_num]['start']:data['entities'][tail_num]['end']])
                rel = i['type']
                relation= '<'+head_enti+','+rel+','+tail_enti+'>; '
                relations += relation
        features.append(relations)
    return features
#print(features)

#根据多样性和复杂性获取的不共享样本，k为每个关系类别选取的数量，rel-dic为经过多样性筛选的句子样本，rel-tensor是计算的句子表示，返回句子文本的list
def get_example(rel_dic:dict,rel_tensor:dict,k):
    result,result_tensor=[],[]
    for i in range(k):
        for key in rel_dic.keys():
            if len(result)==0:
                a=random.randint(0,22)
                result.append(rel_dic[key][a])           
                result_tensor.append(rel_tensor[key][a])                #这个得确认下维度是1,768还是768
                continue
            cos_result=torch.zeros(len(rel_dic[key]))                   #新的类别
            for sent in result_tensor:                 
                cos = torch.nn.functional.cosine_similarity(sent,rel_tensor[key],dim=1)  
                cos_result=cos_result+cos
            index=torch.argmin(cos_result)                              #得是相似性最差的
            result.append(rel_dic[key][index])
    return result
#测试通过

#每个命令重复三次然后选择出现至少两次的三元组,这个通过测试
def gptresult(message):
    concept_test1 = openai.ChatCompletion.create(model="gpt-4",messages=message,temperature=0) 
    concept_test2 = openai.ChatCompletion.create(model="gpt-4",messages=message,temperature=0) 
    concept_test3 = openai.ChatCompletion.create(model="gpt-4",messages=message,temperature=0) 
    answer1=concept_test1['choices'][0]['message']['content']
    answer2=concept_test2['choices'][0]['message']['content']
    answer3=concept_test3['choices'][0]['message']['content']
    answer1_list=answer1.split(';')
    answer2_list=answer1.split(';')
    answer3_list=answer1.split(';')
    #print('answer1:',answer1,'answer2:',answer2,'answer3:',answer3)
    answer=answer1_list+answer2_list+answer3_list
    result=Counter(answer)
    final_result=[]
    for item,count in result.items():
        if count>1:
            final_result.append(item)
    return final_result


def write_result(result,file_name,sent_list):
    with open(file_name,'w') as f:
        for i in range(len(sent_list)):
            print(i)
            for triple in result[i]:
                triple1=triple.replace('<','')
                triple2=triple1.replace('>','')
                temp_data=triple2.split('|')
                f.write(sent_list[i][:-1]+'\t <arg1> '+temp_data[0]+' </arg1> <rel> '+temp_data[1]+' </rel> <arg2> '+temp_data[2]+' </arg2>\t0.1\n')


def get_argument(triple_list):
    arguments=[]
    for triple in triple_list:
        triple1=triple.replace('<','')
        triple2=triple1.replace('>','')
        data=triple2.split('|')
        if data[0] not in arguments:
            arguments.append(data[0])
        if data[2] not in arguments:
            arguments.append(data[2])
    result=filter_argument(arguments)
    return result
                

def filter_argument(arguments):
    result=[]
    for argument in arguments:
        a=argument.split(' ')
        if len(a)>3:
            result.append(argument)
    return result                    #不弄相似度了，就让GPT算把


def find_edges(edge_list,node_num):               #根据边集合和给定的父节点，寻找子节点和孙子数
    child_num,node_list=[],[]                        
    for data in edge_list:
        if node_num==data[0][0] and data[1]!='ROOT':                  #节点找到了，要记录下面的子节点
            node_list.append(Node(data[1]))
            child_num.append(data[0][1])
    if child_num==[]:
        return None,None
    else:
        return node_list,child_num


def get_depencey_tree(tree_edge_list,child_list=None,node_list=None,tree=None):     
    node_num=0
    if child_list==None:
        for data in tree_edge_list:
            if data[1]=='ROOT':
                tree=Node(data[1])
                child_num=data[0][0]
                node_list,child_list=find_edges(tree_edge_list,child_num)     #这样就获得了一开始的child_num和node
                if child_list==None:
                    continue
                node_num+=1
                break                 #有的句子有多个ROOT，我这里选第一个出现的ROOT，也是最大子树
    for i in range(len(child_list)):
        new_node,new_child=find_edges(tree_edge_list,child_list[i])
        if new_node==None:
            tree.addkid(node_list[i])
            node_num+=1
            #print(node_list[i])
        if new_node!=None:                                                    
            node_num+=1
            temp_tree,temp_tree_num=get_depencey_tree(tree_edge_list,child_list=new_child,node_list=new_node,tree=node_list[i])
            node_num+=temp_tree_num
            #如果一个节点有孩子，那么就以它为根，得到带有孩子的子树，然后一起加到原树上，这样能保证节点之间的层级不变
            tree.addkid(temp_tree)
    return tree,node_num
#同样测试成功，明天开始正式做实验！


def get_constituency_tree(tree_data):               #这里tree_data接受的时候就要tree_data[0]
    node_num=0                                      #记录树的总节点数以备用
    for i in range(len(tree_data)):
        if i==0:                                               #反正永远是第一项是标签，那干脆就只取第一项
            temp_node=Node(tree_data[i])
            node_num+=1
        else:                                                  
            if isinstance(tree_data[i],str):                   #除了第一项，只有可能到最底层，其余项也是str，不然别的项都是list，正好递归
                continue
            else:
                temp_node.addkid(get_constituency_tree(tree_data[i])[0])
                node_num+=get_constituency_tree(tree_data[i])[1]
    return temp_node, node_num
#测试成功！NB！

#一个样例，放在这儿
"""old_message=[{"role": "system", "content": "You're a researcher at OpenAI.Users want you to give a brief introduction to OpenAI's products"},{"role": "user", "content": "Give me a brief introduction to the GPT-4 model"}]

concept_test = openai.ChatCompletion.create(
    model="gpt-4",
    messages=old_message,
    temperature=0)
print(concept_test)"""