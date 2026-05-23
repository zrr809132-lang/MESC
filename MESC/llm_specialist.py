"""LLM-based specialist doctors for proposing inquiry symptoms.

Each activated department receives the current patient state, its matched
diseases, and a constrained symptom list, then proposes one symptom to ask.
"""

from specialist_router import route_confidence_to_specialists
from llm_backend import get_model_output_content, find_substr_close_to_end


LLM_SPECIALIST_PROMPT_TEMPLATE = """你是一名{specialist_name}。

当前患者已知症状：
- 存在的症状：{positive_symptoms}
- 不存在的症状：{negative_symptoms}

当前诊断置信度最高的疾病：
{top_diseases_diagnostic_confidence}

你本轮负责鉴别的疾病：
{matched_diseases}

与你负责疾病相关的症状知识如下，数值表示该症状在对应疾病中的历史出现频率：
{specialist_knowledge}

可选择询问的合法候选症状如下：
{candidate_symptoms}

请你像专科医生一样，从候选症状中选择一个最值得下一步询问的症状。优先选择能验证或排除你负责疾病、且对当前 top 疾病有鉴别意义的症状。

输出要求：
- 必须从候选症状中选择一个。
- 结尾必须写成：选择xx作为接下来向患者询问的症状。
请先简要说明理由。"""


class LLMSpecialistAgent:
    def __init__(self, dataset_name, disease_knowledge, symptom_index_dict=None):
        self.dataset_name = dataset_name
        self.disease_knowledge = disease_knowledge
        self.symptom_index_dict = symptom_index_dict or {}

    def _get_empirical_knowledge(self, disease_name):
        disease_info = self.disease_knowledge.get(disease_name, {})
        if "empirical_knowledge" in disease_info:
            return disease_info["empirical_knowledge"]
        return disease_info

    def _is_valid_symptom(self, symptom_name):
        return not self.symptom_index_dict or symptom_name in self.symptom_index_dict

    def build_candidate_symptoms(
        self,
        matched_diseases,
        current_symptom_status,
        max_candidates_per_disease=12,
    ):
        candidate_symptoms = []
        candidate_seen = set()

        for disease_name in matched_diseases:
            empirical_knowledge = self._get_empirical_knowledge(disease_name)
            sorted_symptoms = sorted(
                empirical_knowledge.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            for symptom_name, _ in sorted_symptoms[:max_candidates_per_disease]:
                if symptom_name in current_symptom_status:
                    continue
                if not self._is_valid_symptom(symptom_name):
                    continue
                if symptom_name in candidate_seen:
                    continue
                candidate_seen.add(symptom_name)
                candidate_symptoms.append(symptom_name)

        return candidate_symptoms

    def build_specialist_knowledge(self, matched_diseases, candidate_symptoms):
        specialist_knowledge = {}
        for disease_name in matched_diseases:
            empirical_knowledge = self._get_empirical_knowledge(disease_name)
            specialist_knowledge[disease_name] = {
                symptom: empirical_knowledge.get(symptom, 0.0)
                for symptom in candidate_symptoms
            }
        return specialist_knowledge

    def propose_for_specialist(
        self,
        specialist,
        top_diseases_diagnostic_confidence,
        current_symptom_status,
        llm_name,
        llm,
        tokenizer,
    ):
        matched_diseases = specialist["matched_diseases"]
        candidate_symptoms = self.build_candidate_symptoms(
            matched_diseases=matched_diseases,
            current_symptom_status=current_symptom_status,
        )
        if not candidate_symptoms:
            return None

        positive_symptoms = "、".join(
            symptom for symptom, status in current_symptom_status.items() if status == 1
        ) or "无"
        negative_symptoms = "、".join(
            symptom for symptom, status in current_symptom_status.items() if status == -1
        ) or "无"
        specialist_knowledge = self.build_specialist_knowledge(
            matched_diseases=matched_diseases,
            candidate_symptoms=candidate_symptoms,
        )
        prompt = LLM_SPECIALIST_PROMPT_TEMPLATE.format(
            specialist_name=specialist["name"],
            positive_symptoms=positive_symptoms,
            negative_symptoms=negative_symptoms,
            top_diseases_diagnostic_confidence={
                disease: round(confidence, 3)
                for disease, confidence in top_diseases_diagnostic_confidence.items()
            },
            matched_diseases=matched_diseases,
            specialist_knowledge=specialist_knowledge,
            candidate_symptoms=candidate_symptoms,
        )
        reasoning = get_model_output_content(
            model_name=llm_name,
            model=llm,
            tokenizer=tokenizer,
            prompt=prompt,
            do_sample=False,
            max_new_tokens=512,
        )
        proposed_symptom = find_substr_close_to_end(reasoning, candidate_symptoms)
        selected_by_llm = bool(proposed_symptom)
        if not proposed_symptom:
            proposed_symptom = candidate_symptoms[0]

        return {
            "specialist_id": specialist["id"],
            "specialist_name": specialist["name"],
            "matched_diseases": matched_diseases,
            "candidate_symptoms": candidate_symptoms,
            "proposed_symptom": proposed_symptom,
            "selected_by_llm": selected_by_llm,
            "reasoning": reasoning,
        }

    def propose_by_specialists(
        self,
        diagnostic_confidence,
        current_symptom_status,
        llm_name,
        llm,
        tokenizer,
        top_k_diseases=3,
        max_specialists=3,
    ):
        routing = route_confidence_to_specialists(
            dataset_name=self.dataset_name,
            diagnostic_confidence=diagnostic_confidence,
            top_k=top_k_diseases,
            max_specialists=max_specialists,
        )
        top_diseases = routing["top_diseases"]
        top_diseases_diagnostic_confidence = {
            disease: diagnostic_confidence[disease]
            for disease in top_diseases
            if disease in diagnostic_confidence
        }
        proposals = []

        for specialist in routing["specialists"]:
            proposal = self.propose_for_specialist(
                specialist=specialist,
                top_diseases_diagnostic_confidence=top_diseases_diagnostic_confidence,
                current_symptom_status=current_symptom_status,
                llm_name=llm_name,
                llm=llm,
                tokenizer=tokenizer,
            )
            if proposal is not None:
                proposals.append(proposal)

        flat_candidates = []
        seen_symptoms = set()
        for proposal in proposals:
            symptom_name = proposal["proposed_symptom"]
            if symptom_name in seen_symptoms:
                continue
            seen_symptoms.add(symptom_name)
            flat_candidates.append(
                {
                    "symptom": symptom_name,
                    "specialist_id": proposal["specialist_id"],
                    "specialist_name": proposal["specialist_name"],
                    "matched_diseases": proposal["matched_diseases"],
                    "selected_by_llm": proposal["selected_by_llm"],
                    "reasoning": proposal["reasoning"],
                }
            )

        return {
            "top_diseases": top_diseases,
            "activated_specialists": routing["specialists"],
            "llm_specialist_proposals": proposals,
            "llm_specialist_flat_candidates": flat_candidates,
        }
