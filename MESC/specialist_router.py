from copy import deepcopy


SPECIALIST_PROFILES = {
    "respiratory": {
        "name": "呼吸科医生",
        "focus": "呼吸系统疾病、咳嗽、咳痰、发热、喘息、胸闷气促等症状鉴别",
        "candidate_goal": "优先提出能区分肺炎、支气管哮喘、支气管炎、肺结核等疾病的症状",
    },
    "ent": {
        "name": "耳鼻喉科医生",
        "focus": "鼻炎、咽喉炎、扁桃体炎、外耳炎等耳鼻喉相关疾病鉴别",
        "candidate_goal": "优先提出鼻塞、流涕、打喷嚏、咽部不适、耳痛等相关症状",
    },
    "gastroenterology": {
        "name": "消化科医生",
        "focus": "食管、胃肠、胆胰疾病和腹痛、腹泻、反酸、呕吐、恶心等症状鉴别",
        "candidate_goal": "优先提出反酸、腹痛、腹泻、恶心、呕吐、黑便等消化系统症状",
    },
    "cardiology": {
        "name": "心内科医生",
        "focus": "冠心病、高血压、风湿性心脏病等心血管疾病鉴别",
        "candidate_goal": "优先提出胸痛、胸闷、心悸、活动后加重等心血管相关症状",
    },
    "endocrinology": {
        "name": "内分泌科医生",
        "focus": "甲状腺疾病和代谢内分泌相关表现鉴别",
        "candidate_goal": "优先提出颈部不适、甲状腺肿大、乏力、心悸等症状",
    },
    "general_surgery": {
        "name": "普通外科/消化外科医生",
        "focus": "急性阑尾炎、胆囊结石伴胆囊炎、急性胰腺炎等腹部外科疾病鉴别",
        "candidate_goal": "优先提出右下腹痛、上腹痛、胆绞痛、反跳痛、发热、呕吐等急腹症相关症状",
    },
    "neurology_neurosurgery": {
        "name": "神经/神经外科医生",
        "focus": "脑外伤、头痛、头晕、意识异常、恶心呕吐等神经系统相关鉴别",
        "candidate_goal": "优先提出头痛、头晕、外伤史、意识改变、恶心呕吐等症状",
    },
    "dermatology": {
        "name": "皮肤科医生",
        "focus": "皮炎、皮疹、瘙痒、疱疹等皮肤表现鉴别",
        "candidate_goal": "优先提出皮疹、瘙痒、疱疹、红斑等皮肤相关症状",
    },
    "ophthalmology": {
        "name": "眼科医生",
        "focus": "结膜炎和眼部红肿、分泌物、疼痛、瘙痒等眼部症状鉴别",
        "candidate_goal": "优先提出眼红、眼痒、眼部分泌物、畏光等眼科症状",
    },
    "breast_surgery": {
        "name": "乳腺外科医生",
        "focus": "乳腺炎、乳腺肿瘤和乳房疼痛、红肿、包块等症状鉴别",
        "candidate_goal": "优先提出乳房疼痛、红肿、肿块、发热等乳腺相关症状",
    },
    "urology": {
        "name": "泌尿外科医生",
        "focus": "肾结石、输尿管结石和腰痛、血尿、尿痛等泌尿系统症状鉴别",
        "candidate_goal": "优先提出腰痛、血尿、尿频、尿急、尿痛等泌尿系统症状",
    },
    "pediatric_infectious": {
        "name": "儿科医生",
        "focus": "儿童常见病、儿童感染性疾病、手足口病和小儿腹泻等疾病鉴别",
        "candidate_goal": "优先提出发热、腹泻、皮疹、疱疹、精神状态、脱水表现等儿科相关症状",
    },
    "infectious_tuberculosis": {
        "name": "结核病科/感染科医生",
        "focus": "肺结核和慢性感染性疾病相关症状鉴别",
        "candidate_goal": "优先提出低热、盗汗、咳嗽、咳血、消瘦、乏力等结核相关症状",
    },
}


DATASET_DISEASE_PRIMARY_SPECIALIST_MAP = {
    "GMD": {
        "食管炎": "gastroenterology",
        "肠炎": "gastroenterology",
        "支气管哮喘": "respiratory",
        "冠心病": "cardiology",
        "肺炎": "respiratory",
        "鼻炎": "ent",
        "甲状腺炎": "endocrinology",
        "脑外伤": "neurology_neurosurgery",
        "皮炎": "dermatology",
        "外耳炎": "ent",
        "结膜炎": "ophthalmology",
        "乳腺炎": "breast_surgery",
    },
    "DXY": {
        "过敏性鼻炎": "ent",
        "上呼吸道感染": "respiratory",
        "肺炎": "respiratory",
        "手足口病": "pediatric_infectious",
        "小儿腹泻": "pediatric_infectious",
    },
    "CMD": {
        "细菌性肺炎": "respiratory",
        "间质性肺炎": "respiratory",
        "慢性扁桃体炎": "ent",
        "胃溃疡": "gastroenterology",
        "肾结石": "urology",
        "乳腺恶性肿瘤": "breast_surgery",
        "急性支气管炎": "respiratory",
        "急性胃炎": "gastroenterology",
        "重症肺炎": "respiratory",
        "结节性甲状腺肿": "endocrinology",
        "肺炎": "respiratory",
        "慢性胃炎": "gastroenterology",
        "急性咽炎": "ent",
        "冠心病": "cardiology",
        "手足口病": "pediatric_infectious",
        "急性阑尾炎": "general_surgery",
        "急性胃肠炎": "gastroenterology",
        "输尿管结石": "urology",
        "急性毛细支气管炎": "respiratory",
        "胆囊结石伴胆囊炎": "general_surgery",
        "高血压Ⅲ": "cardiology",
        "支气管哮喘": "respiratory",
        "肺结核": "infectious_tuberculosis",
        "急性胰腺炎": "general_surgery",
        "急性扁桃体炎": "ent",
        "上消化道出血": "gastroenterology",
        "风湿性心脏病": "cardiology",
    },
}

# Backward-compatible alias. Each disease now maps to one primary specialist.
DATASET_DISEASE_SPECIALIST_MAP = DATASET_DISEASE_PRIMARY_SPECIALIST_MAP


def get_specialist_profile(specialist_id):
    """Return one specialist profile by id."""
    profile = SPECIALIST_PROFILES.get(specialist_id)
    return deepcopy(profile) if profile else None


def get_specialists_for_disease(dataset_name, disease_name):
    """Return the primary specialist id as a single-item list for compatibility."""
    specialist_id = DATASET_DISEASE_PRIMARY_SPECIALIST_MAP.get(dataset_name, {}).get(disease_name)
    return [specialist_id] if specialist_id else []


def route_diseases_to_specialists(dataset_name, disease_names, max_specialists=None):
    """Route a disease list to activated specialist profiles.

    The output preserves disease order and avoids duplicate specialists. Each
    activated specialist also records which input diseases triggered it.
    """
    activated = {}

    for disease_name in disease_names:
        for specialist_id in get_specialists_for_disease(dataset_name, disease_name):
            if specialist_id not in activated:
                profile = get_specialist_profile(specialist_id)
                if profile is None:
                    continue
                profile["id"] = specialist_id
                profile["matched_diseases"] = []
                activated[specialist_id] = profile
            activated[specialist_id]["matched_diseases"].append(disease_name)

            if max_specialists is not None and len(activated) >= max_specialists:
                return list(activated.values())

    return list(activated.values())


def get_top_diseases_from_confidence(diagnostic_confidence, top_k=3):
    """Return the top-k disease names from a diagnostic confidence dict."""
    if not diagnostic_confidence:
        return []
    sorted_items = sorted(
        diagnostic_confidence.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    return [disease_name for disease_name, _ in sorted_items[:top_k]]


def route_confidence_to_specialists(
    dataset_name,
    diagnostic_confidence,
    top_k=3,
    max_specialists=None,
):
    """Route current top-k diagnostic diseases to specialist profiles.

    This is the dynamic routing entry point for each inquiry round:
    1. sort diagnostic_confidence by probability;
    2. take the top-k diseases;
    3. activate specialists linked to those diseases.
    """
    top_diseases = get_top_diseases_from_confidence(
        diagnostic_confidence=diagnostic_confidence,
        top_k=top_k,
    )
    specialists = route_diseases_to_specialists(
        dataset_name=dataset_name,
        disease_names=top_diseases,
        max_specialists=max_specialists,
    )
    return {
        "top_diseases": top_diseases,
        "specialists": specialists,
    }


def validate_dataset_specialist_map(dataset_name, disease_names):
    """Find diseases that do not have a specialist mapping."""
    mapped = DATASET_DISEASE_SPECIALIST_MAP.get(dataset_name, {})
    return [disease_name for disease_name in disease_names if disease_name not in mapped]
