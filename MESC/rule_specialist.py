"""Rule-based specialist agents that propose discriminative inquiry symptoms."""

from copy import deepcopy

from specialist_router import route_confidence_to_specialists


class SpecialistAgent:
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

    def score_symptom_for_disease(
        self,
        symptom_name,
        target_disease,
        top_diseases,
        target_weight=0.6,
        gap_weight=0.4,
    ):
        target_knowledge = self._get_empirical_knowledge(target_disease)
        target_prob = float(target_knowledge.get(symptom_name, 0.0))

        other_probs = []
        for disease_name in top_diseases:
            if disease_name == target_disease:
                continue
            disease_knowledge = self._get_empirical_knowledge(disease_name)
            other_probs.append(float(disease_knowledge.get(symptom_name, 0.0)))

        other_max = max(other_probs) if other_probs else 0.0
        positive_gap = target_prob - other_max
        score = target_weight * target_prob + gap_weight * max(positive_gap, 0.0)

        return {
            "symptom": symptom_name,
            "score": round(score, 6),
            "target_prob": round(target_prob, 6),
            "other_max": round(other_max, 6),
            "positive_gap": round(positive_gap, 6),
        }

    def suggest_for_target_disease(
        self,
        target_disease,
        top_diseases,
        current_symptom_status,
        max_candidates=1,
    ):
        target_knowledge = self._get_empirical_knowledge(target_disease)
        scored_symptoms = []

        for symptom_name in target_knowledge:
            if symptom_name in current_symptom_status:
                continue
            if not self._is_valid_symptom(symptom_name):
                continue

            scored_symptoms.append(
                self.score_symptom_for_disease(
                    symptom_name=symptom_name,
                    target_disease=target_disease,
                    top_diseases=top_diseases,
                )
            )

        scored_symptoms.sort(
            key=lambda item: (
                item["score"],
                item["positive_gap"],
                item["target_prob"],
            ),
            reverse=True,
        )
        selected = scored_symptoms[:max_candidates]

        for item in selected:
            item["reason"] = (
                f"该症状在{target_disease}中概率为{item['target_prob']:.3f}，"
                f"相对其他top疾病最大概率差为{item['positive_gap']:.3f}，"
                "具备鉴别价值"
            )

        return selected

    def suggest_candidates(
        self,
        diagnostic_confidence,
        current_symptom_status=None,
        top_k_diseases=3,
        max_specialists=3,
        max_candidates_per_specialist=1,
    ):
        current_symptom_status = current_symptom_status or {}
        routing = route_confidence_to_specialists(
            dataset_name=self.dataset_name,
            diagnostic_confidence=diagnostic_confidence,
            top_k=top_k_diseases,
            max_specialists=max_specialists,
        )
        top_diseases = routing["top_diseases"]
        specialist_outputs = []

        for specialist in routing["specialists"]:
            specialist_result = {
                "specialist_id": specialist["id"],
                "specialist_name": specialist["name"],
                "matched_diseases": specialist["matched_diseases"],
                "candidates": [],
            }

            for target_disease in specialist["matched_diseases"]:
                candidates = self.suggest_for_target_disease(
                    target_disease=target_disease,
                    top_diseases=top_diseases,
                    current_symptom_status=current_symptom_status,
                    max_candidates=max_candidates_per_specialist,
                )
                for candidate in candidates:
                    enriched_candidate = deepcopy(candidate)
                    enriched_candidate["target_disease"] = target_disease
                    enriched_candidate["specialist_id"] = specialist["id"]
                    enriched_candidate["specialist_name"] = specialist["name"]
                    specialist_result["candidates"].append(enriched_candidate)

            if specialist_result["candidates"]:
                specialist_outputs.append(specialist_result)

        flattened_candidates = [
            candidate
            for specialist in specialist_outputs
            for candidate in specialist["candidates"]
        ]

        return {
            "top_diseases": top_diseases,
            "activated_specialists": routing["specialists"],
            "specialist_candidates": specialist_outputs,
            "flat_candidates": flattened_candidates,
        }
