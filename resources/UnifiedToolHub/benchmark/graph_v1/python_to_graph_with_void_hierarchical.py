"""
pip install tree_sitter
pip install tree_sitter_python
https://github.com/tree-sitter/py-tree-sitter
"""
import re

import matplotlib.pyplot as plt
import networkx as nx
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
from collections import defaultdict
# Your Python code
RAW_CODE = '''
from datetime import datetime

# 定义查询条件
plane_departure = "上海"
plane_arrival = "北京"
departure_date = 20231215
train_departure_station = "上海"
train_arrival_station = "北京"
train_class_type = "二等座"
seat_preference = "A"  # 靠窗座位
check_in_date = "2023-12-15"
check_out_date = "2023-12-16"
hotel_location = "北京"

# 搜索飞机
plane_result = Plane.search(plane_departure, plane_arrival, departure_date)

if plane_result:
    # 假设返回的航班信息包含 flight_id
    flight_id = plane_result.get("flight_id")
    if flight_id:
        ret = Plane.book(flight_id)
    else:
        print("未找到符合条件的航班信息。")
else:
    # 如果没有找到飞机，则尝试订火车票
    train_result = Train.book(
        train_id="G123",  # 示例车次，可扩展为动态车次查询
        departure_station=train_departure_station,
        arrival_station=train_arrival_station,
        departure_date=str(departure_date),
        train_class_type=train_class_type,
        position=seat_preference
    )
    if train_result["state"] in [2, 3]:
        ret = {
            "msg": "成功订火车票，但座位未完全满足偏好",
            "data": train_result.get('ticket')   
        }
    elif train_result.get("state") == 1:
        ret = {
            "msg": "未成功订到火车票。",
            "data": 0
        }
    else:
        ret = {
            "msg": "成功订到火车票",
            "data": train_result.get('ticket')
        }

    # 搜索北京的酒店
    hotel_result = Hotel.search(
        location=hotel_location,
        check_in_date=check_in_date,
        check_out_date=check_out_date
    )
    if hotel_result:
        ret = {
            "msg": "成功找到酒店",
            "data": Hotel.book(hotel_result.get('hotel_name'))
        }
    else:
        ret = {
            "msg": "未找到符合条件的酒店",
            "data": 0
        }
    return ret
'''

# TODO: 这里需要增加更多的target_function
target_functions = {
    "Train.book",
    "Plane.search",
    "Plane.book",
    "Weather.get_weather",
    "Hotel.search",
    # "Hotel.basic_info",
    "Hotel.book",
}

LANGUAGE = Language(tspython.language())
parser = Parser(LANGUAGE)


def remove_comments_and_docstrings(source):
    pattern = r'''
        (?P<comments>
            \#.*?$        # Single-line comments
        )
        |
        (?P<docstrings>
            ("""[\s\S]*?""")   # Triple-quoted strings
            |
            ''' + r"""('''[\s\S]*?''')
        )
    """
    regex = re.compile(pattern, re.MULTILINE | re.VERBOSE)
    return regex.sub('', source)


def get_node_text(node, s_code):
    if node is None:
        return ""
    return s_code[node.start_byte:node.end_byte].decode('utf8')


# Function to extract data flow information
def extract_data_flow(node, s_code, assignments, parent_if_condition=None, branch=None):
    if node.type == 'assignment':
        # Handle assignments
        left_node = node.child_by_field_name('left')
        right_node = node.child_by_field_name('right')
        variable_name = get_node_text(left_node, s_code)
        expression = get_node_text(right_node, s_code)
        assignment_info = {
            'variable': variable_name,
            'expression': expression,
            'depends_on': [],
            'line': node.start_point[0] + 1,
            'branch': branch  # Add branch information
        }
        # Find dependencies in the expression
        identifiers = extract_data_flow(right_node, s_code, assignments, branch=branch)

        assignment_info['depends_on'] = identifiers

        # Add parent if condition dependencies
        if parent_if_condition:
            assignment_info['depends_on'].extend(parent_if_condition)
        assignment_info['depends_on'] = set(assignment_info['depends_on'])
        assignments.append(assignment_info)
        return []

    elif node.type == 'call':
        # Only process calls to target functions
        if node.children[0].type == 'attribute':
            function_name = get_node_text(node.children[0], s_code)
            if function_name in target_functions:
                return [function_name]  # Return the target function as a dependency

        return []
    elif node.type == 'subscript':
        # Handle dict[key] access
        obj = get_node_text(node.child_by_field_name('value'), s_code)
        print("subscript", obj)
        return [obj]
    elif node.type == 'if_statement':
        # Handle if statements
        condition_node = node.child_by_field_name('condition')
        condition_identifiers = extract_data_flow(condition_node, s_code, assignments, branch=branch)
        # Process the body of the if statement
        body_node = node.child_by_field_name('consequence')
        for child in body_node.children:
            extract_data_flow(child, s_code, assignments, parent_if_condition=condition_identifiers, branch='True')
        # Process the else clause if it exists
        else_node = node.child_by_field_name('alternative')
        if else_node:
            for child in else_node.children:
                extract_data_flow(child, s_code, assignments, parent_if_condition=condition_identifiers, branch='False')
        return []
    elif node.type == 'identifier':
        return [get_node_text(node, s_code)]
    else:
        # Recursively process child nodes
        ret = []
        for child in node.children:
            ret.extend(extract_data_flow(child, s_code, assignments, parent_if_condition, branch=branch))
        return ret

def merge_nodes(G):
    # 用来存储需要删除的边和节点
    edges_to_remove = []
    nodes_to_remove = []
    
    # 遍历所有边
    for u, v, data in list(G.edges(data=True)): 
        type_u = G.nodes[u].get('type')
        type_v = G.nodes[v].get('type')

        if type_u != 'function' and type_v != 'function':
            continue

        # 合并：如果u是function，v是变量，且u-v是直接连接的，就把v合并到u
        if type_u == 'function' and type_v != 'function':
            neighbors_v = list(G.neighbors(v)) 
            for neighbor in neighbors_v:
                if neighbor != u:  
                    edge_data = G.get_edge_data(v, neighbor)
                    if edge_data:
                        new_weight = edge_data.get('weight', True)
                        if 'weight' in data and data['weight'] != new_weight:
                            new_weight = False
                        if 'weight' in data:
                            current_weight = data['weight']
                            if current_weight != new_weight:
                                new_weight = False 
                        G.add_edge(u, neighbor, branch=new_weight)
            nodes_to_remove.append(v)
            edges_to_remove.append((u, v))

        # 合并：如果v是function，u是变量，且v-u是直接连接的，就把u合并到v
        elif type_u != 'function' and type_v == 'function':
            neighbors_u = list(G.neighbors(u)) 
            for neighbor in neighbors_u:
                if neighbor != v:
                    edge_data = G.get_edge_data(u, neighbor)
                    if edge_data: 
                        new_weight = edge_data.get('weight', True) 
                        if 'weight' in data and data['weight'] != new_weight:
                            new_weight = False
                        if 'weight' in data:
                            current_weight = data['weight']
                            if current_weight != new_weight:
                                new_weight = False 
                        G.add_edge(v, neighbor, weight=new_weight)
            # 标记u节点需要删除
            nodes_to_remove.append(u)
            edges_to_remove.append((u, v))  # 删除u-v的边

    # 删除标记的边和节点
    G.remove_edges_from(edges_to_remove)
    G.remove_nodes_from(nodes_to_remove)

def modify_graph(graph):
    void_counter = 0
    for node in list(graph.nodes):
        # 获取当前节点的所有出边
        out_edges = list(graph.out_edges(node, data=True))
        
        # 根据 branch 属性对边分组
        branch_groups = {}
        for _, target, attrs in out_edges:
            branch_value = attrs['branch']
            if branch_value not in branch_groups:
                branch_groups[branch_value] = []
            branch_groups[branch_value].append(target)
        
        # 检查每组是否有两个及以上的边
        for branch_value, targets in branch_groups.items():
            if len(targets) > 1:
                # 创建一个新的 void 节点
                void_node = f"VOID"
                void_counter += 1
                graph.add_node(void_node, type='void')
                
                # 将原始节点连接到 void 节点
                graph.add_edge(node, void_node, branch=branch_value)
                
                # 将 void 节点连接到所有目标节点
                for target in targets:
                    graph.remove_edge(node, target)  # 删除原始边
                    graph.add_edge(void_node, target, branch=branch_value)


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
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'branch'), font_color='red')
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

def try_for_python():
    # Remove comments and docstrings
    s_code = remove_comments_and_docstrings(RAW_CODE)

    # Parse the code
    s_code = bytes(s_code, 'utf8')
    tree = parser.parse(s_code)
    root_node = tree.root_node

    # Data flow structures
    assignments = []
    dict_accesses = []

    # Start extracting data flow information
    extract_data_flow(root_node, s_code, assignments)

    # Print the results
    print("Assignments:")
    for assign in assignments:
        if assign['depends_on']:
            print(f"{assign['variable']}  Depends on: {assign['depends_on']}")

    G = nx.DiGraph()
    node_info = {}
    function_info = {}

    # Ensure only target functions are added as nodes
    for assign in assignments:
        var = assign['variable']
        use_function = None
        for function in target_functions:
            if function in assign['depends_on']:
                use_function = function
        if use_function:
            node_info[var] = {
                "function": use_function,
            }
            function_info[use_function] = var
            G.add_node(use_function, type='function')

    print("Nodes in the graph after adding:")
    for node, data in G.nodes(data=True):
        print(f"Node: {node}, Attributes: {data}")

    # Create edges only between functions, avoiding variables like 'flight_id'
    for assign in assignments:
        use_function = None
        for function in target_functions:
            if function in assign['depends_on']:
                use_function = function
        var = assign['variable']
        depends_on = assign['depends_on']
        
        # Only add edges between functions, not variables
        if len(depends_on) > 0:
            if use_function:
                var = use_function
            if var != "ret":
                for dep in depends_on:
                    if dep != var:
                        # Skip variables like flight_id
                        if dep in node_info:
                            dep = node_info[dep]["function"]
                        # Add edge between function nodes
                        G.add_edge(dep, var, branch=assign.get('branch', ""))

    print("Nodes in the graph after adding, again:")
    for node, data in G.nodes(data=True):
        print(f"Node: {node}, Attributes: {data}")
    # 合并node
    merge_nodes(G)
    print(f"answer for merge: ")
    print(G.edges(data = True))
    print(G.nodes(data=True))

    # 加入虚空节点
    modify_graph(G)

    # 处理成分层图
    # node_levels = calculate_node_levels(G)
    # draw_hierarchical_graph(G, node_levels)

    # 可视化图形(非分层图)
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
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'branch'), font_color='red')
    # nx.draw_networkx_labels(G, pos, font_size=8)
    # plt.title('Data Flow Graph')
    # plt.axis('off')
    # plt.show()



if __name__ == "__main__":
    # Your Python code
    RAW_CODE = '''
    from datetime import datetime

    # 定义查询条件
    plane_departure = "上海"
    plane_arrival = "北京"
    departure_date = 20231215
    train_departure_station = "上海"
    train_arrival_station = "北京"
    train_class_type = "二等座"
    seat_preference = "A"  # 靠窗座位
    check_in_date = "2023-12-15"
    check_out_date = "2023-12-16"
    hotel_location = "北京"

    # 搜索飞机
    plane_result = Plane.search(plane_departure, plane_arrival, departure_date)

    if plane_result:
        # 假设返回的航班信息包含 flight_id
        flight_id = plane_result.get("flight_id")
        if flight_id:
            ret = Plane.book(flight_id)
        else:
            print("未找到符合条件的航班信息。")
    else:
        # 如果没有找到飞机，则尝试订火车票
        train_result = Train.book(
            train_id="G123",  # 示例车次，可扩展为动态车次查询
            departure_station=train_departure_station,
            arrival_station=train_arrival_station,
            departure_date=str(departure_date),
            train_class_type=train_class_type,
            position=seat_preference
        )
        if train_result["state"] in [2, 3]:
            ret = {
                "msg": "成功订火车票，但座位未完全满足偏好",
                "data": train_result.get('ticket')   
            }
        elif train_result.get("state") == 1:
            ret = {
                "msg": "未成功订到火车票。",
                "data": 0
            }
        else:
            ret = {
                "msg": "成功订到火车票",
                "data": train_result.get('ticket')
            }

        # 搜索北京的酒店
        hotel_result = Hotel.search(
            location=hotel_location,
            check_in_date=check_in_date,
            check_out_date=check_out_date
        )
        if hotel_result:
            ret = {
                "msg": "成功找到酒店",
                "data": Hotel.book(hotel_result.get('hotel_name'))
            }
        else:
            ret = {
                "msg": "未找到符合条件的酒店",
                "data": 0
            }
        return ret
    '''
    try_for_python(RAW_CODE)
