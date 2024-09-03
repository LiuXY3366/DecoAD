from py2neo import Graph, Node, NodeMatcher, Relationship, RelationshipMatcher


# 创建连接
def connect(url='http://localhost:7474', username='neo4j',password='neo4j1234'):
    ##连接neo4j数据库，输入地址、用户名、密码
    graph = Graph(url, name='neo4j',auth=(username, password))
    return graph

def create_node(label,name):
    graph = connect()
    node = Node(label,label = label,name=name)
    graph.create(node)
    print(f'添加成功:{node}')
    return node

def create_relation(label1,label2,name1,name2,relation):
    graph = connect()
    node1 = search_node(label1,name1).first()
    node2 = search_node(label2,name2).first()
    if node1 and node2:
        # 创建关系并将其连接到现有节点
        node2node = Relationship(node1,relation,node2)
        graph.create(node2node)
    else:
        print("未找到一个或两个现有节点")

def search_relation(label1,label2,name1,name2):
    graph = connect()
    node_matcher = NodeMatcher(graph)
    relationship_matcher = RelationshipMatcher(graph)

    node1 = search_node(label1,name1).first()
    node2 = search_node(label2,name2).first()

    if node1 and node2:
        # 查找两个节点之间的关系
        relationships = relationship_matcher.match(nodes=(node1, node2))
        for relationship in relationships:
            relationship = str(relationship)  # 将关系转换为字符串
            if "abnormal" in relationship:
                return 'abnormal'
            elif "normal" in relationship:
                return 'normal'
            else:
                # print(f'{name1}和{name2}没有关系！')
                return 'no_relation'
    else:
        print("未找到一个或两个节点")
        return None


# 删除制定节点，只删除一个没有关系节点
def delete_node_single(label,name):
    graph = connect()
    matcher = NodeMatcher(graph)  # 创建关系需要用到
    # 删除指定节点，只删除一个没有关系节点
    node = matcher.match(label).where(name=name).first()  # 先匹配，叫黄帝内经的第一个结点
    graph.delete(node)

# 删除制定关系和节点
def delete_node_relation(label,name):
    graph = connect()
    matcher = NodeMatcher(graph)  # 创建关系需要用到
    nr = list(matcher.match(label).where(name=name))[0]
    graph.delete(nr)

# 查找所有节点
def search_all_node():
    graph = connect()
    for node in graph.nodes:
        print(graph.nodes[node])


# 查询指定节点
def search_node(label=None,name=None):
    graph = connect()
    n = graph.nodes.match(label,label=label,name=name)
    return n

# 查找所有关系
def search_all_relation():
    graph = connect()
    rps = graph.relationships
    for r in rps:
        print(rps[r])
    print("======================")
    return rps

# 删除有关name的节点和关系
def delete_all_node_relation(label,name):
    graph = connect()
    try:
        graph.run('match(n:'+label+'label{name:'+name+'}) detach delete n')
        print('删除成功')
    except:
        print('删除失败')

# 转化为numpy数据
def search_label_to_numpy(label):
    graph = connect()
    # 转化为numpy数据
    sa = graph.run('match(n:'+label+') return *').to_ndarray()
    print(sa)
    return sa

# 以字典列表的形式返回查询的结果数据
def search_label_to_dict(label):
    graph = connect()
    # 以字典列表的形式返回查询的结果数据
    sa = graph.run('match(n:'+label+') return *').data()
    print(sa)
    return sa

def init_anything(scene=29,pose=38,flag =''):
    scene_arr = [1, 2, 3, 13, 14, 29, 31, 35, 36, 38, 42, 43, 47, 48, 54, 55, 68, 76, 77, 92, 94, 99, 109, 111, 121,
                 122, 124, 127, 129, 148, 149, 150, 151, 154, 155, 158, 164, 235, 236, 248, 268, 273, 282]
    if scene == 29:
        for i in range(scene):
            create_node('scene', f'scene{i+1}')
    elif flag == 'NWPUC':
        for i in range(len(scene_arr)):
            create_node('scene',f'scene{scene_arr[i]}')
    else:
        for i in range(scene):
            create_node('scene', f'scene{i+1}')


    for i in range(pose):
        print(f'pose{i+1}')
        create_node('pose',f'pose{i+1}')

# 清除一切，慎用！
def clean_all():
    graph = connect()
    # 编写Cypher查询来删除所有节点和关系
    cypher_query = "MATCH (n) DETACH DELETE n"

    # 执行查询
    graph.run(cypher_query)

    print('清除一切！')

# 清除没有建立关系的结点！
def clean_no_relation():
    graph = connect()
    # 编写Cypher查询来删除所有节点和关系
    # cypher_query = "MATCH (n) DETACH DELETE n"
    cypher_query = "MATCH (n) WHERE NOT (n)--() DETACH DELETE n"

    # 执行查询
    graph.run(cypher_query)

    print('清除没有建立关系的结点！')



def main():
    clean_all()
    init_anything(scene=29,pose=100)
    return





if __name__ == '__main__':
    # main()
    # search_all_relation()
    search_all_node()