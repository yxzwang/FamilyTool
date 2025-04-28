import networkx as nx
import re
import matplotlib.pyplot as plt
from collections import defaultdict
# TODO: 多个IF的改写

def parse_expression(expression):
    """将input变为list"""
    tokens = re.findall(r'[A-Za-z0-9_.]+|THEN|IF|ELSE|AND', expression)
    return tokens

def check_format(input:list):
    found_A = False
    found_B = False
    for item in input:
        if item == "ELSE":
            return found_A and found_B
        if item == "IF":
            found_A = True
        if item == "THEN":
            found_B = True
    return False

def build_graph(input: list):
    # 格式检查，ELSE一定在IF和THEN后面
    else_id = 0
    if "ELSE" in input:
        check = check_format(input)
        else_id = input.index("ELSE")
        if not check:
            return False
        
    # 开始进行建图
    G = nx.DiGraph()
    VOID_NODE = "VOID"
    
    # 找到IF对应的then_id -> 暂时只考虑只有一个IF的情况
    then_id_list = []
    found_if = False
    for item in input:
        if item == 'IF':
            found_if = True
            if_id = input.index(item)
        # 找到对应的第一个THEN
        elif found_if and item == "THEN":
            then_id= input.index(item)
            then_id_list.append([if_id, then_id])
            break
    # 格式检查：IF之后一定有THEN，否则错误退出
    if found_if and not then_id:
        return False
    
    logic = {"IF", "THEN", "ELSE", "AND"} 
    for item in input:
        if item not in logic:
            G.add_node(item)

    # 第一步：处理THEN的连接，并进行节点合并
    print(f"input before then is: {input}")
    # while "THEN" in input and input.index("THEN") != then_id_list:
    while "THEN" in input and input.index("THEN") != then_id:
        i = input.index("THEN")
        G.add_edge(input[i-1], input[i+1], description="True")
        input.pop(i)
        input.pop(i)
    i = 0
    while 1:
        if i < len(input):
            if input[i] =="THEN" and i != then_id:
                G.add_edge(input[i-1], input[i+1], description="True")
                input.pop(i)
                input.pop(i)
            else:
                i += 2
        else:
            break

    
    # TODO: 如果出现"C THEN A AND B"，应该将C和B进行and连接，而不是A和B -> 找到如果前面有连接的连接
    # 第二步：处理AND的连接，创建AND块作为一个整体
    # 对于所有AND，进行图节点的连接
    # 如果有很多AND应该怎么办 -> VOID_NODE应该改名字
    print(f"input before and is: {input}")
    while "AND" in input:
        i = input.index("AND")
        G.add_node(VOID_NODE)
        G.add_edge(VOID_NODE, input[i+1], description="AND")
        G.add_edge(VOID_NODE, input[i-1], description="AND")
        
        k = 2
        # 处理连续的AND
        while i + k < len(input) and input[i + k] == 'AND':
            G.add_edge(VOID_NODE, input[i + 1 + k], description="AND")
            k += 2
        # 删除AND部分的元素，从后往前删除，避免索引错乱
        for j in range(i + k - 1, i - 2, -1):
            input.pop(j)
        input.insert(i, VOID_NODE)



    # 第三步：处理IF THEN ELSE的连接
    # TODO: "IF C THEN A AND B"会出现IF-THEN连接冗余，先这里的逻辑应该还可以改
    # 这里先假设只出现一次IF，并且一定是放在最前面的
    for i in range(len(input)):
        if input[i] == "IF":
            G.add_edge(input[i+1], input[then_id+1], description="True")
            if "ELSE" in input:
                else_id = input.index("ELSE")
                G.add_edge(input[i+1], input[else_id+1], description="False")
    for node in list(G.nodes):
        if node.startswith("VOID"):
            predecessors = list(G.predecessors(node))
            if predecessors:
                pred_description = G.edges[predecessors[0], node].get("description", "False")
                for successor in G.successors(node):
                    G.edges[node, successor]["description"] = pred_description
    # 可视化图
    print(G.edges(data=True))
    print(G.nodes(data=True))
    return G
def calculate_node_levels(G):
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("The graph must be a Directed Acyclic Graph (DAG).")
    
    # Perform topological sort
    topo_sorted = list(nx.topological_sort(G))
    
    # Dictionary to store the level of each node
    node_levels = {}
    
    # Initialize levels for all nodes
    for node in topo_sorted:
        # The level of a node is one more than the maximum level of its predecessors
        if G.in_edges(node):
            node_levels[node] = max(node_levels[predecessor] for predecessor in G.predecessors(node)) + 1
        else:
            # If no predecessors, assign level 0
            node_levels[node] = 0
    
    return node_levels

def draw_hierarchical_graph(G, node_levels):
    # Create positions for nodes
    pos = {}
    level_counts = defaultdict(int)  # Keep track of node count in each level for horizontal placement
    
    for node, level in node_levels.items():
        # Place nodes horizontally spaced within their level
        pos[node] = (level_counts[level], -level)  # (x, y)
        level_counts[level] += 1
    
    # Draw the graph
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'description'), font_color='red')
    plt.title("Hierarchical Graph")
    plt.show()

    # plt.figure(figsize=(12, 8))
    # nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100)
    # nx.draw_networkx_edges(G, pos, arrows=True)  # 设置 arrows=True 显示箭头
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'branch'), font_color='red')
    # nx.draw_networkx_labels(G, pos, font_size=8)
    # plt.title('Data Flow Graph')
    # plt.axis('off')
    # plt.show()

def try_for_dsl(expression):
    input = parse_expression(expression)
    G = build_graph(input)
    return G


if __name__ == "__main__":
    expression_1 = "IF Plane.search THEN Plane.book ELSE Train.search AND Hotel.search THEN Hotel.book"
    # expression_1 = "C THEN A AND B"
    # expression_1 = "IF C THEN A AND B"
    input_1 = parse_expression(expression_1)
    G = build_graph(input_1)

    # 处理成分层图
    # node_levels = calculate_node_levels(G)
    # draw_hierarchical_graph(G, node_levels)


    # 可视化图形
    # pos = nx.spring_layout(G)
    # node_colors = []
    # for node in G.nodes(data=True):
    #     if node[1].get('type') == 'variable':
    #         node_colors.append('grey')
    #     else:
    #         node_colors.append('lightblue')

    # plt.figure(figsize=(12, 8))
    # nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100)
    # nx.draw_networkx_edges(G, pos, arrows=True)  # 设置 arrows=True 显示箭头
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'description'), font_color='red')
    # nx.draw_networkx_labels(G, pos, font_size=8)
    # plt.title('Data Flow Graph')
    # plt.axis('off')
    # plt.show()
