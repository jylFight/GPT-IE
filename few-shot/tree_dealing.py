from zss import simple_distance, Node


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
    if child_list==None:
        for data in tree_edge_list:
            if data[1]=='ROOT':
                tree=Node(data[1])
                child_num=data[0][0]
                node_list,child_list=find_edges(tree_edge_list,child_num)     #这样就获得了一开始的child_num和node
    
    for i in range(len(child_list)):
        new_node,new_child=find_edges(tree_edge_list,child_list[i])
        if new_node==None:
            tree.addkid(node_list[i])
            #print(node_list[i])
        if new_node!=None:                                                    
            temp_tree=get_depencey_tree(tree_edge_list,child_list=new_child,node_list=new_node,tree=node_list[i])
            #如果一个节点有孩子，那么就以它为根，得到带有孩子的子树，然后一起加到原树上，这样能保证节点之间的层级不变
            tree.addkid(temp_tree)
    return tree
#测试成功!!

def get_constituency_tree(tree_data):                          #这里tree_data接受的时候就要tree_data[0]
    node_num=0                                                 #记录树的总节点数以备用
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
    return (temp_node,node_num)
#测试成功！NB！