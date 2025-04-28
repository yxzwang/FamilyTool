import keyword
import re
from .python_to_graph_with_void_hierarchical import try_for_python
from .dsl_to_graph_with_void_hierarchical_more_if import try_for_dsl

def detect_code_type(input_code):
    """
    判断输入是 Python 代码还是 DSL
    :param input_code: str 输入的代码块
    :return: "Python" 或 "DSL"
    """
    # 检查 Python 的特征
    python_keywords = set(keyword.kwlist)
    python_features = [":", "def ", "return", "import", "#", "class ", "from "]

    # 检查是否包含 Python 关键字
    if any(kw in input_code for kw in python_keywords):
        return "Python"
    
    # 检查 Python 特有的语法
    if any(feature in input_code for feature in python_features):
        return "Python"

    # 检查 DSL 的特征
    dsl_keywords = {"IF", "THEN", "ELSE", "AND"}
    tokens = re.findall(r'[A-Za-z0-9_.]+|THEN|IF|ELSE|AND', input_code)
    if any(token in dsl_keywords for token in tokens):
        return "DSL"

    # 如果无法明确判断，返回未知类型
    return "Unknown"

def compare_graphs(G1, G2):
    alpha = 0.5 # 节点相似度权重
    beta = 0.5  # 边相似度权重
    # 节点相似度
    node_G1 = set(G1.nodes())
    node_G2 = set(G2.nodes())
    common_nodes = node_G1.intersection(node_G2)    # 交
    total_nodes = node_G1.union(node_G2)    # 并
    node_similarity = len(common_nodes) / len(total_nodes) if total_nodes else 1.0

    # 边相似度
    edge_G1 = set((u, v, frozenset(data.items())) for u, v, data in G1.edges(data=True))
    edge_G2 = set((u, v, frozenset(data.items())) for u, v, data in G2.edges(data=True))
    # edge_G1 = set(G1.edges(data=True))
    # edge_G2 = set(G2.edges(data=True))
    common_edges = edge_G1.intersection(edge_G2)    # 交
    total_edges = edge_G1.union(edge_G2)    # 并
    edge_similarity = len(common_edges) / len(total_edges) if total_edges else 1.0

    accuracy = alpha * node_similarity + beta * edge_similarity
    return accuracy



if __name__ == "__main__":
    input_text_1 = ""
    input_text_2 = ""
    type_1 = detect_code_type(input_text_1)
    type_2 = detect_code_type(input_text_2)
    #  对input_text_1
    if type_1 == "DSL":
        G1 = try_for_dsl(input_text_1)
    elif type_1 == "Python":
        G1 = try_for_python(input_text_1)
    else:
        print(f"Fail to recognize type for input_text_1")
    #  对input_text_2
    if type_2 == "DSL":
        G2 = try_for_dsl(input_text_2)
    elif type_2 == "Python":
        G2 = try_for_python(input_text_2)
    else:
        print(f"Fail to recognize type for input_text_2")
    ans = compare_graphs(G1, G2)
    print(ans)
