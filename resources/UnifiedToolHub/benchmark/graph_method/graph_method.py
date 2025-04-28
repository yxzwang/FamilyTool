import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_v1.dsl_to_graph_with_void import build_graph, parse_expression
from graph_v1.dsl_to_graph_with_void_hierarchical_more_if import calculate_node_levels, draw_hierarchical_graph
import networkx as nx

import matplotlib.pyplot as plt

def draw_graph(G):
    # 处理成分层图
    node_levels = calculate_node_levels(G)
    draw_hierarchical_graph(G, node_levels)

def method_v1(groundtruth_input, model_input):
    expression_1 = parse_expression(groundtruth_input)
    expression_2 = parse_expression(model_input)

    # G1：标准答案图，G2：模型预测图
    G1 = build_graph(expression_1)
    G2 = build_graph(expression_2)

    # # 可视化图
    # draw_graph(G1)
    # draw_graph(G2)
    
    alpha = 0.5 # 节点相似度权重
    beta = 0.5  # 边相似度权重
    node_similarity = 0.0   # 顶点相似度
    edge_similarity = 0.0   # 边相似度

    node_G1 = set(G1.nodes())
    node_G1.discard("VOID")     # 计算指标时不考虑VOID节点
    node_G2 = set(G2.nodes())
    node_G2.discard("VOID")
    print(f"node_G1: {node_G1}")
    print(f"node_G2: {node_G2}")

    common_nodes = node_G1.intersection(node_G2)    # 交
    total_nodes = node_G1.union(node_G2)    # 并
    print(f"common_nodes: {common_nodes}")
    print(f"total_nodes: {total_nodes}")
    node_similarity = len(common_nodes) / len(total_nodes) if total_nodes else 1.0
    print(f"node_similarity: {node_similarity}")

    edge_G1 = set((u, v, frozenset(data.items())) for u, v, data in G1.edges(data=True))
    edge_G2 = set((u, v, frozenset(data.items())) for u, v, data in G2.edges(data=True))

    common_edges = edge_G1.intersection(edge_G2)    # 交
    total_edges = edge_G1.union(edge_G2)    # 并
    print(f"common_edges: {common_edges}")
    print(f"total_edges: {total_edges}")
    edge_similarity = len(common_edges) / len(total_edges) if total_edges else 1.0
    print(f"edge_similarity: {edge_similarity}")

    score = alpha * node_similarity + beta * edge_similarity
    return score

# 获取每个节点的前驱节点
def get_predecessors(G):
    predecessors = {}
    for node in G.nodes():
        predecessors[node] = list(G.predecessors(node))
    return predecessors

# 获取拓扑排序
def topological_sort(G):
    return list(nx.topological_sort(G))

def method_v2(groundtruth_input, model_input):
    expression_1 = parse_expression(groundtruth_input)
    expression_2 = parse_expression(model_input)

    # G1：标准答案图，G2：模型预测图
    G1 = build_graph(expression_1)
    G2 = build_graph(expression_2)

    # # 可视化图
    # draw_graph(G1)
    # draw_graph(G2)

    # 获取拓扑排序，用于计算C(v)
    topo_G1 = topological_sort(G1)
    topo_G2 = topological_sort(G2)

    node_G1 = set(G1.nodes())
    node_G1.discard("VOID")     # 计算指标时不考虑VOID节点
    node_G2 = set(G2.nodes())
    node_G2.discard("VOID")

    C = {}  # 正确性指示函数
    predecessors_G1 = get_predecessors(G1)  # 获取G1的前驱节点

    for v in topo_G1:
        # 检查当前节点是否在模型预测的图中
        if v not in topo_G2:
            C[v] = 0
            continue
        # 检查是否有前驱节点
        if not list(G1.predecessors(v)):
            C[v] = 1
        else:
            valid = 1
            for u in predecessors_G1[v]:
                # 检查前驱节点是否在模型预测的图中
                if C[u] != 1:
                    valid = 0
                    break
                edge = (u, v)
                if edge not in G2.edges():
                    valid = 0
                    break
                if G1.edges[edge] != G2.edges[edge]:
                    valid = 0
                    break
            C[v] = valid
    C.pop("VOID", None)
    penalty = len(node_G1.intersection(node_G2)) / len(node_G1.union(node_G2))
    score = sum(C.values()) / len(node_G1) * penalty
    return score

if __name__ == "__main__":
    
    groundtruth_input = "IF Plane.search THEN Plane.book ELSE Train.search THEN Train.book AND Hotel.search THEN Hotel.book"
    model_input = "IF Plane.search THEN Plane.book ELSE Train.search AND Hotel.book"


    print(f"score: {method_v1(groundtruth_input, model_input)}")
    print(f"score: {method_v2(groundtruth_input, model_input)}")

