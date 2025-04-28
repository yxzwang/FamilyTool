# README.md

## 概述

本项目包含三个 Python 脚本，用于将 Python 和 DSL 语句转换为包含虚空节点的图结构，同时支持分层图的输出，并对生成的图进行比较。

### 脚本功能说明

1. **`python_to_graph_with_void_hierarchical.py`**  
   - 将 Python 语句转换为包含虚空节点的图。  
   - 输出分层图结构。  

2. **`dsl_to_graph_with_void_hierarchical.py`**  
   - 将 DSL 语句转换为包含虚空节点的图。  
   - 输出分层图结构。  

3. **`compare_graphs_v1.py`**  
   - 判断输入是 DSL 还是 Python。  
   - 计算两段输入生成的图的相似度。  

---

### 图相似度计算_v1

相似度的计算方式如下：

#### 公式：
accuracy = alpha * 节点相似度 + beta * 边相似度

#### 参数：
- **`alpha`**：节点相似度的权重（默认值：0.5）。  
- **`beta`**：边相似度的权重（默认值：0.5）。  

#### 相似度计算规则：
- **节点相似度**：`gold_plan` 和 `model_plan` 节点集合的交集除以并集。  
- **边相似度**：`gold_plan` 和 `model_plan` 边集合（包括边的 `"description"` 相同的边）的交集除以并集。  
  - *注意*：考虑到 `model_plan` 可能包含额外信息，不使用 `gold_plan` 的召回率作为衡量标准。  

---

### DSL 逻辑范式优先级

对于 DSL 中的三种逻辑范式，优先级如下：
1. **`THEN`**（最高优先级）  
2. **`AND`**  
3. **`IF...THEN...ELSE`（或 `IF...THEN...`）**（最低优先级）  

---

### TODO 列表

1. **支持 DSL 的多 `IF` 场景**  

2. **改进图中边的细节处理**  
   - 探索更细粒度的边属性继承（例如具体槽位级别的继承）。  

3. **自动扩展 `target_function` 的覆盖范围，提升图生成的准确性。**  

4. **开发针对 `golden_answer` 为图结构时的处理方法**  

5. **支持 alpha 和 beta 的自动调节**  

6. **先跑一些数据看看效果评测**

---

### 使用说明

1. **图生成**：
   - 对于 Python 输入，运行 `python_to_graph_with_void_hierarchical.py`。  
   - 对于 DSL 输入，运行 `python_to_graph_with_void_dsl.py`。

2. **图比较**：
   - 使用 `compare_graphs_v1.py` 对两段输入生成的图计算相似度。  

   示例：
   ```bash
   python compare_graphs_v1.py
