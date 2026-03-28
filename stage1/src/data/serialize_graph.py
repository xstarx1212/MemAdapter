"""
图序列化和验证工具
"""
import re
from typing import Dict, List, Set, Tuple


class GraphParser:
    """解析和验证图的序列化格式"""
    
    @staticmethod
    def parse_graph(graph_text: str) -> Dict:
        """
        解析图文本为结构化格式
        
        返回:
        {
            "nodes": {node_id: node_text, ...},
            "edges": [(src, dst, relation), ...]
        }
        """
        result = {
            "nodes": {},
            "edges": []
        }
        
        # 查找 <NODES> 和 <EDGES> 部分
        nodes_match = re.search(r'<NODES>(.*?)(?=<EDGES>|$)', graph_text, re.DOTALL)
        edges_match = re.search(r'<EDGES>(.*?)(?=$|\n\[)', graph_text, re.DOTALL)
        
        # 解析节点
        if nodes_match:
            nodes_text = nodes_match.group(1).strip()
            for line in nodes_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # 格式: N1: entity text
                match = re.match(r'(N\d+):\s*(.+)', line)
                if match:
                    node_id, node_text = match.groups()
                    result["nodes"][node_id] = node_text
        
        # 解析边
        if edges_match:
            edges_text = edges_match.group(1).strip()
            for line in edges_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # 支持两种格式:
                # 1. 新格式: E1: N1 -> N2
                # 2. 旧格式: N1 -> N2: relation
                match = re.match(r'E\d+:\s*(N\d+)\s*->\s*(N\d+)', line)
                if match:
                    # 新格式（带边 ID）
                    src, dst = match.groups()
                    relation = ""  # 新格式不包含关系描述
                    result["edges"].append((src, dst, relation))
                else:
                    # 旧格式（带关系描述）
                    match = re.match(r'(N\d+)\s*->\s*(N\d+)(?::\s*(.+))?', line)
                    if match:
                        src, dst, relation = match.groups()
                        relation = relation or ""
                        result["edges"].append((src, dst, relation))
        
        return result
    
    @staticmethod
    def validate_graph(full_graph: Dict, subgraph: Dict) -> Tuple[bool, List[str]]:
        """
        验证子图是否为全图的合法子集
        
        返回: (is_valid, error_messages)
        """
        errors = []
        
        # 检查子图节点是否在全图中
        full_nodes = set(full_graph["nodes"].keys())
        sub_nodes = set(subgraph["nodes"].keys())
        
        invalid_nodes = sub_nodes - full_nodes
        if invalid_nodes:
            errors.append(f"子图包含全图中不存在的节点: {invalid_nodes}")
        
        # 检查子图边是否在全图中
        # 论文要求子图是 strict subset：边需要同时满足 (src, dst, relation) 都在 full graph 中。
        full_edges = set((src, dst, (rel or "").strip()) for src, dst, rel in full_graph["edges"])
        
        for src, dst, relation in subgraph["edges"]:
            # 检查边的节点是否存在
            if src not in full_nodes:
                errors.append(f"边 {src} -> {dst} 的源节点不在全图中")
            if dst not in full_nodes:
                errors.append(f"边 {src} -> {dst} 的目标节点不在全图中")
            
            # 检查边是否在全图中（含 relation）
            rel_norm = (relation or "").strip()
            if (src, dst, rel_norm) not in full_edges:
                errors.append(f"边 {src} -> {dst}: {rel_norm} 不在全图的边集中")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def format_graph(nodes: Dict[str, str], edges: List[Tuple[str, str, str]]) -> str:
        """
        将结构化图格式化为文本
        """
        lines = ["<NODES>"]
        for node_id in sorted(nodes.keys()):
            lines.append(f"{node_id}: {nodes[node_id]}")
        
        lines.append("")
        lines.append("<EDGES>")
        for src, dst, relation in edges:
            if relation:
                lines.append(f"{src} -> {dst}: {relation}")
            else:
                lines.append(f"{src} -> {dst}")
        
        return "\n".join(lines)


def validate_teacher_output(gold_full_graph: str, gold_subgraph: str) -> Tuple[bool, List[str]]:
    """验证教师输出的图是否合法"""
    parser = GraphParser()
    
    try:
        # 解析
        full = parser.parse_graph(gold_full_graph)
        sub = parser.parse_graph(gold_subgraph)
        
        # 验证
        is_valid, errors = parser.validate_graph(full, sub)
        
        return is_valid, errors
        
    except Exception as e:
        return False, [f"解析错误: {str(e)}"]


if __name__ == "__main__":
    # 测试
    test_full_graph = """[FULL_GRAPH]
<NODES>
N1: Albert Einstein
N2: Theory of Relativity
N3: Nobel Prize
N4: Physics

<EDGES>
N1 -> N2: proposed
N1 -> N3: awarded
N2 -> N4: field_of
N3 -> N4: in_field
"""
    
    test_subgraph = """[EVIDENCE_SUBGRAPH]
<NODES>
N1: Albert Einstein
N2: Theory of Relativity

<EDGES>
N1 -> N2: proposed
"""
    
    parser = GraphParser()
    full = parser.parse_graph(test_full_graph)
    sub = parser.parse_graph(test_subgraph)
    
    print("Full graph nodes:", full["nodes"])
    print("Full graph edges:", full["edges"])
    print()
    print("Subgraph nodes:", sub["nodes"])
    print("Subgraph edges:", sub["edges"])
    print()
    
    is_valid, errors = parser.validate_graph(full, sub)
    print(f"Valid: {is_valid}")
    if errors:
        print("Errors:", errors)
