import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class MemoryAgent:
    def __init__(self, golden_records_path):
        """
        初始化记忆中枢
        golden_records_path: 提取好的历史高分病历 JSON 文件路径
        """
        import json

        with open(golden_records_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        self.texts = [record["semantic_query"] for record in self.records]

        print("[Memory Agent] 正在加载 Embedding 模型...")
        
        self.embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")

        print("[Memory Agent] 正在向量化历史经验...")
        self.embeddings = self.embedder.encode(self.texts, normalize_embeddings=True)

        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(np.array(self.embeddings).astype("float32"))
        print(f"[Memory Agent] 记忆库已就绪，容量: {self.index.ntotal} 条。")

    def _build_query_text(self, current_symptoms_dict):
        positive_symptoms = [k for k, v in current_symptoms_dict.items() if v == 1]
        negative_symptoms = [k for k, v in current_symptoms_dict.items() if v == -1]

        if not positive_symptoms and not negative_symptoms:
            return ""

        positive_text = "、".join(positive_symptoms)
        negative_text = "、".join([f"无{symptom}" for symptom in negative_symptoms])

        if positive_text and negative_text:
            return f"患者有{positive_text}；{negative_text}"
        if positive_text:
            return positive_text
        return negative_text

    def retrieve_best_match(self, current_symptoms_dict):
        """
        根据新患者的症状字典，检索最相似的历史病例
        """
        query_text = self._build_query_text(current_symptoms_dict)
        if not query_text:
            return None  

        query_vec = self.embedder.encode(
            [query_text], normalize_embeddings=True
        ).astype("float32")
        similarities, indices = self.index.search(query_vec, 1)  

        sim_score = similarities[0][0]
        best_match_index = int(indices[0][0])
        best_match = self.records[best_match_index]
        return {
            "match_index": best_match_index,
            "score": sim_score,
            "disease": best_match["disease"],
            "trigger_positive": best_match["trigger_positive"],
            "trigger_negative": best_match["trigger_negative"],
            "winning_path": [item["symptom"] for item in best_match["golden_path"]],
            "path_details": best_match["golden_path"],
            "max_steps": best_match["max_steps"],
            "query_text": query_text,
            "matched_text": best_match["semantic_query"],
        }

    def retrieve_winning_path(self, current_symptoms_dict, threshold=0.88):
        """
        根据新患者的症状字典，检索历史最优提问路径
        """
        best_match = self.retrieve_best_match(current_symptoms_dict)
        if best_match is None:
            return None

        sim_score = best_match["score"]
        if sim_score >= threshold:
            return best_match
        return None
