from stable_baselines3.common.callbacks import BaseCallback
import torch as th
import os
import json
import time
from inquiry_policy import SymptomInquiryActorCriticPolicy
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from reproducibility import set_global_seed
from sklearn.metrics import f1_score
from llm_backend import get_llm_usage, reset_llm_usage

try:
    from case_memory import MemoryAgent
except ImportError:
    MemoryAgent = None


_MEMORY_AGENT = None
_MEMORY_AGENT_DISABLED = False
MEMORY_THRESHOLD = 0.80
MEMORY_TRIGGER_BUDGET = 3
COMPACT_SPC_INTERACTIONS = True
_MEMORY_AGENT_PATH = None

# 得到标签疾病的排名
def get_disease_label_rank(diagnostic_confidence: dict, disease_label: str):
    sorted_diagnostic_confidence = dict(sorted(diagnostic_confidence.items(), key=lambda item: item[1], reverse=True))  # 将诊断置信度按从高到低排序
    rank = 0
    pre_confidence = 1
    for disease, confidence in sorted_diagnostic_confidence.items():
        if confidence != pre_confidence:
            pre_confidence = confidence
            rank += 1
        if disease == disease_label:
            return rank
    return 999


def get_memory_agent(golden_records_path):
    global _MEMORY_AGENT, _MEMORY_AGENT_DISABLED, _MEMORY_AGENT_PATH

    if _MEMORY_AGENT_DISABLED:
        return None
    if _MEMORY_AGENT is not None and _MEMORY_AGENT_PATH == golden_records_path:
        return _MEMORY_AGENT
    if MemoryAgent is None:
        print("[Memory Agent] 未找到可用依赖，跳过记忆拦截。")
        _MEMORY_AGENT_DISABLED = True
        return None

    if not os.path.exists(golden_records_path):
        print(f"[Memory Agent] 未找到 {golden_records_path}，跳过记忆拦截。")
        _MEMORY_AGENT_DISABLED = True
        return None

    try:
        _MEMORY_AGENT = MemoryAgent(golden_records_path)
        _MEMORY_AGENT_PATH = golden_records_path
    except Exception as exc:
        print(f"[Memory Agent] 初始化失败，跳过记忆拦截: {exc}")
        _MEMORY_AGENT_DISABLED = True
        return None
    return _MEMORY_AGENT


def build_memory_interaction(interaction, memory_match, memory_path, phase):
    memory_interaction = deepcopy(interaction)
    if phase == "memory_trigger_check":
        question_stage = "trigger_completion"
        question_stage_desc = "补问相似黄金病例中尚未确认的 trigger 阳性/阴性症状"
    else:
        question_stage = "golden_path"
        question_stage_desc = "执行复核通过后的黄金路径核心问题"

    memory_interaction["source_agent"] = "memory_agent"
    memory_interaction["question_source"] = "memory_agent"
    memory_interaction["question_stage"] = question_stage
    memory_interaction["question_stage_desc"] = question_stage_desc
    memory_interaction["memory_replay"] = True
    memory_interaction["memory_phase"] = phase
    memory_interaction["memory_score"] = round(float(memory_match["score"]), 4)
    memory_interaction["memory_disease"] = memory_match["disease"]
    memory_interaction["memory_path"] = memory_path
    memory_interaction["memory_matched_text"] = memory_match["matched_text"]
    return memory_interaction


def get_known_symptom_status(env):
    return {
        symptom: status
        for symptom, status in env.current_symptom_status.items()
        if status in (1, -1)
    }


def build_memory_handoff_context(
    handoff_type,
    memory_match,
    env,
    reason,
    trigger_interactions=None,
    confirmed_match=None,
):
    trigger_interactions = trigger_interactions or []
    trigger_answers = [
        {
            "symptom": interaction.get("asked_symptom"),
            "answer": interaction.get("symptom_status"),
        }
        for interaction in trigger_interactions
        if interaction.get("asked_symptom") is not None
    ]
    return {
        "handoff_type": handoff_type,
        "rejected_memory_disease": memory_match.get("disease"),
        "match_index": memory_match.get("match_index"),
        "initial_score": round(float(memory_match.get("score", 0.0)), 4),
        "confirmed_score": (
            round(float(confirmed_match["score"]), 4) if confirmed_match else None
        ),
        "matched_text": memory_match.get("matched_text"),
        "rejected_reason": reason,
        "trigger_answers": trigger_answers,
        "confirmed_positive": [
            item["symptom"] for item in trigger_answers if item.get("answer") == 1
        ],
        "confirmed_negative": [
            item["symptom"] for item in trigger_answers if item.get("answer") == -1
        ],
        "known_symptom_status": get_known_symptom_status(env),
    }


def ask_memory_questions(env, symptom_names, memory_match, phase):
    asked_interactions = []
    terminal_env_info = None
    terminated = False
    truncated = False

    for symptom_name in symptom_names:
        if symptom_name in env.current_symptom_status:
            continue

        action_idx = env.symptom2idx.get(symptom_name)
        if action_idx is None:
            continue

        _, _, terminated, truncated, env_info = env.step(action_idx)
        terminal_env_info = env_info
        interaction = build_memory_interaction(
            interaction=env_info["interactions"][-1],
            memory_match=memory_match,
            memory_path=symptom_names,
            phase=phase,
        )
        interaction["asked_symptom"] = symptom_name
        interaction["known_symptom_status_after_step"] = get_known_symptom_status(env)
        asked_interactions.append(interaction)

        if terminated or truncated:
            break

    return asked_interactions, terminal_env_info, terminated, truncated


def refresh_memory_final_diagnosis(env, interactions):
    current_obs = env.get_observation()
    if current_obs is None:
        return

    symptom_status_sub_state = env.update_current_symptom_status()
    diagnostic_confidence_sub_state = env.update_diagnostic_confidence()
    env.state = np.concatenate([symptom_status_sub_state, diagnostic_confidence_sub_state])
    env.final_diagnostic_confidence = env.diagnostic_confidence.copy()
    if interactions:
        interactions[-1]["diagnostic_confidence"] = env.final_diagnostic_confidence.copy()


def get_top_diagnostic_confidence(diagnostic_confidence, top_k=5):
    return {
        disease: round(float(confidence), 6)
        for disease, confidence in list(diagnostic_confidence.items())[:top_k]
    }


def normalize_diagnostic_confidence(diagnostic_confidence):
    return {
        disease: float(confidence)
        for disease, confidence in diagnostic_confidence.items()
    }


def summarize_activated_specialists(activated_specialists):
    return [
        {
            "specialist_name": specialist.get("name"),
            "specialist_id": specialist.get("id"),
            "matched_diseases": specialist.get("matched_diseases", []),
        }
        for specialist in activated_specialists or []
    ]


def summarize_llm_specialist_proposals(proposals):
    return [
        {
            "specialist_name": proposal.get("specialist_name"),
            "specialist_id": proposal.get("specialist_id"),
            "matched_diseases": proposal.get("matched_diseases", []),
            "proposed_symptom": proposal.get("proposed_symptom"),
            "selected_by_llm": proposal.get("selected_by_llm"),
            "reasoning_summary": (proposal.get("reasoning") or "").strip()[-240:],
        }
        for proposal in proposals or []
    ]


def build_compact_spc_interaction(policy, env_idx, env_info, batch_metadata=None, selected_symptom=None):
    batch_metadata = batch_metadata or {}
    decision_info = policy.decision_info[env_idx]
    raw_interaction = env_info["interactions"][-1]
    diagnostic_confidence = raw_interaction.get("diagnostic_confidence") or {}
    known_symptom_status = raw_interaction.get("known_symptom_status_after_step") or {}
    selected_symptom = selected_symptom or decision_info["selected_symptom"]

    return {
        "source_agent": "spc",
        "question_source": "llm_specialist_batch",
        "question_stage": "specialist_batch_inquiry",
        "question_stage_desc": "LLM 专科医生各提出 1 个症状，系统按批次逐个询问并更新诊断置信度",
        **batch_metadata,
        "selected_symptom": selected_symptom,
        "symptom_status": raw_interaction.get("symptom_status"),
        "answer_text": "有" if raw_interaction.get("symptom_status") == 1 else "没有",
        "is_recorded": raw_interaction.get("is_recorded"),
        "diagnostic_confidence": normalize_diagnostic_confidence(diagnostic_confidence),
        "diagnostic_top1": next(iter(diagnostic_confidence), None),
        "known_symptom_status_after_step": known_symptom_status,
        "specialist_candidate_source": decision_info.get("specialist_candidate_source"),
        "activated_specialists_summary": summarize_activated_specialists(
            decision_info.get("activated_specialists")
        ),
        "llm_specialist_proposals_summary": summarize_llm_specialist_proposals(
            decision_info.get("llm_specialist_proposals")
        ),
        "specialist_batch_symptoms": decision_info.get("specialist_batch_symptoms"),
        "memory_handoff_summary": (
            {
                "handoff_type": decision_info["memory_handoff_context"].get("handoff_type"),
                "rejected_memory_disease": decision_info["memory_handoff_context"].get("rejected_memory_disease"),
                "initial_score": decision_info["memory_handoff_context"].get("initial_score"),
                "rejected_reason": decision_info["memory_handoff_context"].get("rejected_reason"),
            }
            if decision_info.get("memory_handoff_context")
            else None
        ),
    }


def build_spc_interaction(policy, env_idx, env_info, batch_metadata=None, selected_symptom=None):
    if COMPACT_SPC_INTERACTIONS:
        return build_compact_spc_interaction(
            policy=policy,
            env_idx=env_idx,
            env_info=env_info,
            batch_metadata=batch_metadata,
            selected_symptom=selected_symptom,
        )

    batch_metadata = batch_metadata or {}
    decision_info = policy.decision_info[env_idx]
    interaction = {
        "source_agent": "spc",
        "question_source": "specialist_agent_batch",
        "question_stage": "specialist_batch_inquiry",
        "question_stage_desc": "专科医生生成候选症状，系统跳过 LLM 症状筛选并按候选顺序批量询问；每个症状逐个更新诊断置信度",
        "retry_history": decision_info["retry_history"],
        "candidate_symptoms": decision_info["candidate_symptoms"],
        "candidate_symptoms_before_rerank": decision_info.get("candidate_symptoms_before_rerank"),
        "memory_rerank_scores": decision_info.get("memory_rerank_scores"),
        "memory_handoff_context": decision_info.get("memory_handoff_context"),
        "candidate_generation_mode": decision_info.get("candidate_generation_mode"),
        "rl_candidate_symptoms_ignored": decision_info.get("rl_candidate_symptoms_ignored"),
        "candidate_symptoms_before_specialist": decision_info.get("candidate_symptoms_before_specialist"),
        "specialist_augmented_candidates": decision_info.get("specialist_augmented_candidates"),
        "specialist_only_candidates": decision_info.get("specialist_only_candidates"),
        "specialist_batch_symptoms": decision_info.get("specialist_batch_symptoms"),
        "specialist_batch_size_planned": decision_info.get("specialist_batch_size_planned", 0),
        "specialist_top_diseases": decision_info.get("specialist_top_diseases"),
        "activated_specialists": decision_info.get("activated_specialists"),
        "specialist_candidates": decision_info.get("specialist_candidates"),
        "specialist_flat_candidates": decision_info.get("specialist_flat_candidates"),
        "specialist_candidate_source": decision_info.get("specialist_candidate_source"),
        "llm_specialist_used": decision_info.get("llm_specialist_used"),
        "llm_specialist_proposals": decision_info.get("llm_specialist_proposals"),
        "llm_specialist_flat_candidates": decision_info.get("llm_specialist_flat_candidates"),
        "selection_reasoning": decision_info["selection_reasoning"],
        "selected_symptom": selected_symptom or decision_info["selected_symptom"],
        **batch_metadata,
        **env_info["interactions"][-1],
    }
    return interaction


def apply_memory_shortcuts(observations, eval_envs, results, progress_bar, memory_agent, interactions_list):
    if memory_agent is None:
        return observations, False

    intercepted_any = False

    for env_idx in range(eval_envs.num_envs):
        env = eval_envs.envs[env_idx]

        if getattr(env, "is_completed", False) or env.is_eval_env_completed():
            continue

        if getattr(env, "memory_checked", False):
            continue

        current_symptoms = env.current_symptom_status
        best_memory_match = memory_agent.retrieve_best_match(current_symptoms)
        env.memory_checked = True
        if best_memory_match is None:
            print(f"🧪 [环境 {env_idx}] 当前无可检索的阳性症状，切换到 SPC 专科问诊流程。")
            continue

        print(f"🧪 [环境 {env_idx}] 最高相似病例分数: {best_memory_match['score']:.3f}")
        memory_match = (
            best_memory_match if best_memory_match["score"] > MEMORY_THRESHOLD else None
        )

        if not memory_match:
            env.memory_handoff_context = build_memory_handoff_context(
                handoff_type="initial_reject",
                memory_match=best_memory_match,
                env=env,
                reason=f"初始相似度未超过阈值 {MEMORY_THRESHOLD:.2f}",
            )
            print(
                f"↪ [环境 {env_idx}] 未达记忆阈值 > {MEMORY_THRESHOLD:.2f}，切换到 SPC 专科问诊流程。"
            )
            continue

        print(f"💡 [环境 {env_idx}] 命中初步记忆，相似度: {memory_match['score']:.3f}")

        trigger_questions = []
        for symptom_name in memory_match["trigger_positive"]:
            if symptom_name not in env.current_symptom_status:
                trigger_questions.append(symptom_name)
        for symptom_name in memory_match["trigger_negative"]:
            if symptom_name not in env.current_symptom_status:
                trigger_questions.append(symptom_name)
        trigger_questions = trigger_questions[:MEMORY_TRIGGER_BUDGET]

        trigger_interactions = []
        terminal_env_info = None
        terminated = False
        truncated = False

        if trigger_questions:
            print(
                f"🔎 [环境 {env_idx}] 先补问 trigger 症状: {', '.join(trigger_questions)}"
            )
            (
                trigger_interactions,
                terminal_env_info,
                terminated,
                truncated,
            ) = ask_memory_questions(
                env=env,
                symptom_names=trigger_questions,
                memory_match=memory_match,
                phase="memory_trigger_check",
            )
            observations[env_idx] = env.get_observation()

        if terminated or truncated:
            refresh_memory_final_diagnosis(env, trigger_interactions)
            record = {
                "case_route": "memory_agent_trigger_only",
                "case_route_desc": "Memory Agent 命中后只执行 trigger 补问，环境在进入黄金路径前结束",
                "memory_handoff_type": None,
                "used_memory_agent": True,
                "used_spc": False,
                "memory_handoff_used": False,
                "disease_label": terminal_env_info["disease_label"],
                "initial_symptom_status": terminal_env_info["initial_symptom_status"],
                "initial_diagnostic_confidence": terminal_env_info["initial_diagnostic_confidence"],
                "interactions": trigger_interactions,
                "final_diagnostic_confidence": env.final_diagnostic_confidence,
                "final_known_symptom_status": get_known_symptom_status(env),
                "memory_hit": {
                    "score": round(float(memory_match["score"]), 4),
                    "disease": memory_match["disease"],
                    "winning_path": trigger_questions,
                    "memory_confirmed": False,
                },
            }
            results["records"].append(record)
            progress_bar.update(1)
            intercepted_any = True
            interactions_list[env_idx] = []
            if not env.is_eval_env_completed():
                next_observation, _ = env.reset()
                observations[env_idx] = next_observation
            continue

        confirmed_match = memory_agent.retrieve_best_match(env.current_symptom_status)
        if confirmed_match is None:
            interactions_list[env_idx].extend(trigger_interactions)
            env.memory_handoff_context = build_memory_handoff_context(
                handoff_type="trigger_reject",
                memory_match=memory_match,
                env=env,
                reason="trigger 补问后无有效记忆匹配",
                trigger_interactions=trigger_interactions,
            )
            print(f"↪ [环境 {env_idx}] trigger 补问后无有效记忆匹配，切换到 SPC 专科问诊流程。")
            continue

        initial_memory_match = memory_match
        same_record = confirmed_match["match_index"] == initial_memory_match["match_index"]
        score_passed = confirmed_match["score"] > MEMORY_THRESHOLD
        memory_switched = score_passed and not same_record
        confirmed = score_passed
        print(
            f"🧪 [环境 {env_idx}] trigger 补问后最高相似病例分数: {confirmed_match['score']:.3f}"
        )
        if not confirmed:
            interactions_list[env_idx].extend(trigger_interactions)
            env.memory_handoff_context = build_memory_handoff_context(
                handoff_type="trigger_reject",
                memory_match=memory_match,
                env=env,
                reason=(
                    "trigger 复核未通过："
                    f"{'匹配到了不同黄金病例且相似度未超过阈值' if not same_record else '相似度未超过阈值'}"
                ),
                trigger_interactions=trigger_interactions,
                confirmed_match=confirmed_match,
            )
            print(
                f"↪ [环境 {env_idx}] trigger 复核未通过，切换到 SPC 专科问诊流程。"
            )
            continue

        memory_match = confirmed_match
        if memory_switched:
            print(
                f"🔁 [环境 {env_idx}] trigger 复核切换黄金病例: "
                f"{initial_memory_match['disease']}#{initial_memory_match['match_index']} "
                f"→ {memory_match['disease']}#{memory_match['match_index']}，执行新黄金路径。"
            )
        else:
            print(f"✅ [环境 {env_idx}] trigger 复核通过，继续执行黄金路径。")

        core_questions = memory_match["winning_path"][: memory_match["max_steps"]]
        memory_interactions = list(trigger_interactions)

        (
            replay_interactions,
            terminal_env_info,
            terminated,
            truncated,
        ) = ask_memory_questions(
            env=env,
            symptom_names=core_questions,
            memory_match=memory_match,
            phase="memory_golden_path",
        )
        memory_interactions.extend(replay_interactions)

        refresh_memory_final_diagnosis(env, memory_interactions)

        if not env.is_completed:
            _, _, _, _, terminal_env_info = env.step(env.symptom_num)
            if terminal_env_info["interactions"]:
                terminal_env_info["interactions"][-1]["diagnostic_confidence"] = env.final_diagnostic_confidence

        env.is_completed = True

        record = {
            "case_route": "memory_agent",
            "case_route_desc": (
                "Memory Agent 初始命中后，trigger 复核切换到新的黄金病例并执行新黄金路径"
                if memory_switched
                else "Memory Agent 复核通过，直接执行黄金路径完成问诊"
            ),
            "memory_handoff_type": None,
            "used_memory_agent": True,
            "used_spc": False,
            "memory_handoff_used": False,
            "disease_label": terminal_env_info["disease_label"],
            "initial_symptom_status": terminal_env_info["initial_symptom_status"],
            "initial_diagnostic_confidence": terminal_env_info["initial_diagnostic_confidence"],
            "interactions": memory_interactions,
            "final_diagnostic_confidence": env.final_diagnostic_confidence,
            "final_known_symptom_status": get_known_symptom_status(env),
            "memory_hit": {
                "score": round(float(memory_match["score"]), 4),
                "disease": memory_match["disease"],
                "winning_path": core_questions,
                "trigger_questions": trigger_questions,
                "memory_confirmed": True,
                "memory_switched": memory_switched,
                "initial_match": {
                    "match_index": initial_memory_match["match_index"],
                    "score": round(float(initial_memory_match["score"]), 4),
                    "disease": initial_memory_match["disease"],
                    "matched_text": initial_memory_match["matched_text"],
                    "winning_path": initial_memory_match["winning_path"],
                },
                "confirmed_match": {
                    "match_index": memory_match["match_index"],
                    "score": round(float(memory_match["score"]), 4),
                    "disease": memory_match["disease"],
                    "matched_text": memory_match["matched_text"],
                    "winning_path": memory_match["winning_path"],
                },
            },
        }
        results["records"].append(record)
        progress_bar.update(1)
        intercepted_any = True
        interactions_list[env_idx] = []

        if not env.is_eval_env_completed():
            next_observation, _ = env.reset()
            observations[env_idx] = next_observation

    return observations, intercepted_any

# 评估
def performance_eval(llm_name, dataset_name, exp_name, stage, timestep, eval_envs, policy, settings={}):  # 验证和测试时仍采用多环境设置
    results = {
        "timestep": timestep,
        "metrics": {
            "Acc_wo_iq": 0,
            "Acc": 0,
            "Acc_gain": 0,
            "Avg_n": 0,
            "Avg_consultation_rounds": 0,
            "Memory_direct_acc": 0,
            "Handoff_to_SPC_acc": 0,
            "Runtime_seconds": 0,
            "Avg_runtime_per_case_seconds": 0,
            "Avg_LLM_calls_per_case": 0,
            "Avg_tokens_per_case": 0,
        },
        "records": []
    }
    start_time = time.time()
    reset_llm_usage()
    # 重置数据索引和计数器
    for env in eval_envs.envs:
        env.sample_idx = -1
        env.data_count = 0
    # 推理
    with th.no_grad():  # 推理时不计算梯度
        progress_bar = tqdm(total=len(eval_envs.envs[0].dataset), desc=f"{'验证' if stage == 'dev' else '测试'}-{timestep}")
        memory_agent = None
        if stage == "test":
            memory_path = f"./outputs/policy/{dataset_name}/curstom_rl_training/Golden_records.json"
            if os.path.exists(memory_path):
                memory_agent = get_memory_agent(memory_path)
                if memory_agent is not None:
                    print(f"🧠 成功加载 {dataset_name} 专属记忆中枢！")
            else:
                print(f"⚠️ 未找到 {dataset_name} 的 Golden_records.json，跳过直觉拦截。")
        observations = eval_envs.reset()
        all_completed = False
        interactions_list = [[] for _ in eval_envs.envs]
        batch_round_counters = [0 for _ in eval_envs.envs]
        while not all_completed:
            observations, memory_records_added = apply_memory_shortcuts(
                observations=observations,
                eval_envs=eval_envs,
                results=results,
                progress_bar=progress_bar,
                memory_agent=memory_agent,
                interactions_list=interactions_list,
            )
            all_completed = all([eval_env.is_eval_env_completed() for eval_env in eval_envs.envs])
            if memory_records_added:
                save_results(llm_name, dataset_name, exp_name, stage, timestep, results, settings)
                if all_completed:
                    break
                continue
            if all_completed:
                break

            observations = th.from_numpy(observations).float().to(policy.device)
            actions, values, log_probs = policy.forward(obs=observations, exp_name=exp_name, stage=stage)  # policy_info_list记录的是每轮的动作选择过程
            observations, rewards, done_list, env_info_list  = eval_envs.step(actions)  # env_info_list记录的是当前所有轮次的状态
            manual_done_list = [False for _ in eval_envs.envs]
            for i in range(len(eval_envs.envs)):
                batch_symptoms = policy.decision_info[i].get("specialist_batch_symptoms") or []
                is_batch_inquiry = stage == "test" and len(batch_symptoms) > 0
                batch_metadata = {}
                if is_batch_inquiry:
                    batch_round_counters[i] += 1
                    batch_metadata = {
                        "batch_question": True,
                        "batch_round_id": batch_round_counters[i],
                        "batch_position": 1,
                        "batch_size_planned": len(batch_symptoms),
                        "batch_symptoms": batch_symptoms,
                    }

                if not done_list[i]:
                    interactions_list[i].append(
                        build_spc_interaction(
                            policy=policy,
                            env_idx=i,
                            env_info=env_info_list[i],
                            batch_metadata=batch_metadata,
                            selected_symptom=batch_symptoms[0] if is_batch_inquiry else None,
                        )
                    )

                    if is_batch_inquiry:
                        for batch_position, symptom_name in enumerate(batch_symptoms[1:], start=2):
                            env = eval_envs.envs[i]
                            if env.is_eval_env_completed() or getattr(env, "is_completed", False):
                                break
                            if env.current_turn >= env.max_turns:
                                break
                            if symptom_name in env.current_symptom_status:
                                continue

                            action_idx = env.symptom2idx.get(symptom_name)
                            if action_idx is None:
                                continue

                            next_observation, _, terminated, truncated, next_env_info = env.step(action_idx)
                            observations[i] = next_observation
                            extra_batch_metadata = {
                                "batch_question": True,
                                "batch_round_id": batch_round_counters[i],
                                "batch_position": batch_position,
                                "batch_size_planned": len(batch_symptoms),
                                "batch_symptoms": batch_symptoms,
                            }
                            interactions_list[i].append(
                                build_spc_interaction(
                                    policy=policy,
                                    env_idx=i,
                                    env_info=next_env_info,
                                    batch_metadata=extra_batch_metadata,
                                    selected_symptom=symptom_name,
                                )
                            )

                            if terminated or truncated:
                                done_list[i] = True
                                env_info_list[i] = next_env_info
                                manual_done_list[i] = True
                                break
            for i, done in enumerate(done_list):
                if done:
                    current_handoff_context = policy.decision_info[i].get("memory_handoff_context")
                    memory_handoff_used = (
                        current_handoff_context is not None
                        or any(
                            interaction.get("memory_handoff_context") is not None
                            for interaction in interactions_list[i]
                        )
                    )
                    used_memory_agent = (
                        memory_handoff_used
                        or any(
                            interaction.get("source_agent") == "memory_agent"
                            for interaction in interactions_list[i]
                        )
                    )
                    memory_handoff_type = None
                    if memory_handoff_used:
                        handoff_context = current_handoff_context
                        if handoff_context is None:
                            handoff_context = next(
                                (
                                    interaction.get("memory_handoff_context")
                                    for interaction in interactions_list[i]
                                    if interaction.get("memory_handoff_context") is not None
                                ),
                                None,
                            )
                        memory_handoff_type = handoff_context.get("handoff_type") if handoff_context else None

                    case_route = "memory_handoff_to_spc" if memory_handoff_used else "spc"
                    if memory_handoff_type == "trigger_reject":
                        case_route_desc = "Memory Agent 初始命中，但 trigger 补问后复核失败，随后交接给 SPC 专科问诊完成问诊"
                    elif memory_handoff_type == "initial_reject":
                        case_route_desc = "Memory Agent 初始相似度未超过阈值，随后交接给 SPC 专科问诊完成问诊"
                    else:
                        case_route_desc = "未使用 Memory Agent 接管，完整走 SPC 专科问诊流程"

                    new_record = {
                        "case_route": case_route,
                        "case_route_desc": case_route_desc,
                        "memory_handoff_type": memory_handoff_type,
                        "used_memory_agent": used_memory_agent,
                        "used_spc": True,
                        "memory_handoff_used": memory_handoff_used,
                        "disease_label": env_info_list[i]["disease_label"],
                        "initial_symptom_status": env_info_list[i]["initial_symptom_status"],
                        "initial_diagnostic_confidence": env_info_list[i]["initial_diagnostic_confidence"],
                        "interactions": interactions_list[i],
                        "final_known_symptom_status": env_info_list[i].get("final_known_symptom_status", {})
                    }
                    results["records"].append(new_record)
                    interactions_list[i] = []
                    progress_bar.update(1)
                    if manual_done_list[i] and not eval_envs.envs[i].is_eval_env_completed():
                        next_observation, _ = eval_envs.envs[i].reset()
                        observations[i] = next_observation
            if any(done_list):
                save_results(llm_name, dataset_name, exp_name, stage, timestep, results, settings)  # 保存最新结果
            all_completed = all([eval_env.is_eval_env_completed() for eval_env in eval_envs.envs])  # 指示是否所有环境都模拟完了分配的数据
        progress_bar.close()
    # 计算评估指标
    metrics = calculate_metrics(results["records"])
    results["metrics"]["Acc_wo_iq"] = metrics[0]
    results["metrics"]["Acc"] = metrics[1]
    results["metrics"]["Acc_gain"] = round(metrics[1] - metrics[0], 3)
    results["metrics"]["Avg_n"] = metrics[2]
    results["metrics"]["Avg_consultation_rounds"] = metrics[3]
    route_metrics = calculate_route_metrics(results["records"])
    results["metrics"]["Memory_direct_acc"] = route_metrics["Memory_direct_acc"]
    results["metrics"]["Handoff_to_SPC_acc"] = route_metrics["Handoff_to_SPC_acc"]
    elapsed_time = time.time() - start_time
    case_count = max(len(results["records"]), 1)
    llm_usage = get_llm_usage()
    results["metrics"]["Runtime_seconds"] = round(elapsed_time, 3)
    results["metrics"]["Avg_runtime_per_case_seconds"] = round(
        elapsed_time / case_count,
        3,
    )
    results["metrics"]["Avg_LLM_calls_per_case"] = round(
        llm_usage["calls"] / case_count,
        3,
    )
    results["metrics"]["Avg_tokens_per_case"] = round(
        llm_usage["total_tokens"] / case_count,
        1,
    )
    results["metrics"]["LLM_usage"] = llm_usage
    save_results(llm_name, dataset_name, exp_name, stage, timestep, results, settings)  # 保存当前结果
    return results["metrics"]


def count_consultation_rounds(interactions):
    consultation_rounds = 0
    seen_batch_rounds = set()

    for interaction in interactions:
        if interaction.get("batch_question") and interaction.get("batch_round_id") is not None:
            batch_key = (
                interaction.get("question_stage"),
                interaction.get("batch_round_id"),
            )
            if batch_key in seen_batch_rounds:
                continue
            seen_batch_rounds.add(batch_key)
            consultation_rounds += 1
        else:
            consultation_rounds += 1

    return consultation_rounds
        
def calculate_metrics(records):
    correct_wo_iq_count = 0
    correct_count = 0
    turn_count = 0
    consultation_round_count = 0
    for record in records:
        disease_label = record["disease_label"]
        initial_diagnostic_confidence = record["initial_diagnostic_confidence"]
        final_interaction = record["interactions"][-1] if len(record["interactions"]) > 0 else {}
        final_diagnostic_confidence = (
            final_interaction.get("diagnostic_confidence")
            or initial_diagnostic_confidence
        )
        is_initial_diagnosis_correct = get_disease_label_rank(initial_diagnostic_confidence, disease_label) == 1
        is_final_diagnosis_correct = get_disease_label_rank(final_diagnostic_confidence, disease_label) == 1
        correct_wo_iq_count += 1 if is_initial_diagnosis_correct else 0
        correct_count += 1 if is_final_diagnosis_correct else 0
        turn_count += len(record["interactions"])
        consultation_round_count += count_consultation_rounds(record["interactions"])
    records_size = len(records)
    return (
        round(correct_wo_iq_count / records_size, 3),
        round(correct_count / records_size, 3),
        round(turn_count / records_size, 1),
        round(consultation_round_count / records_size, 1),
    )  # Acc_wo_iq, Acc, Avg_n, Avg_consultation_rounds


def get_final_diagnostic_confidence(record):
    for interaction in reversed(record.get("interactions", [])):
        diagnostic_confidence = interaction.get("diagnostic_confidence")
        if diagnostic_confidence:
            return diagnostic_confidence
    return (
        record.get("final_diagnostic_confidence")
        or record.get("initial_diagnostic_confidence")
        or {}
    )


def calculate_route_accuracy(records):
    if not records:
        return 0
    correct_count = 0
    for record in records:
        disease_label = record["disease_label"]
        final_diagnostic_confidence = get_final_diagnostic_confidence(record)
        is_correct = (
            get_disease_label_rank(final_diagnostic_confidence, disease_label) == 1
        )
        correct_count += 1 if is_correct else 0
    return round(correct_count / len(records), 3)


def calculate_route_metrics(records):
    memory_direct_records = [
        record
        for record in records
        if record.get("case_route") == "memory_agent"
    ]
    handoff_to_spc_records = [
        record
        for record in records
        if record.get("case_route") == "memory_handoff_to_spc"
    ]
    return {
        "Memory_direct_acc": calculate_route_accuracy(memory_direct_records),
        "Handoff_to_SPC_acc": calculate_route_accuracy(handoff_to_spc_records),
    }

def calculate_metrics_offline(results_filepath, dgit=26):
    with open(results_filepath, "r", encoding="utf-8") as result_file:
        records = json.load(result_file)["records"]
        
    correct_wo_iq_count = 0
    correct_count = 0
    turn_count = 0
    consultation_round_count = 0
    k = 12
    top_k_counts = [0] * k
    for record in records:
        disease_label = record["disease_label"]
        initial_diagnostic_confidence = record["initial_diagnostic_confidence"]
        initial_diagnostic_confidence = {k: round(v, dgit) for k, v in initial_diagnostic_confidence.items()}
        final_interaction = record["interactions"][-1] if len(record["interactions"]) > 0 else {}
        final_diagnostic_confidence = (
            final_interaction.get("diagnostic_confidence")
            or initial_diagnostic_confidence
        )
        final_diagnostic_confidence = {k: round(v, dgit) for k, v in final_diagnostic_confidence.items()}
        is_initial_diagnosis_correct = get_disease_label_rank(initial_diagnostic_confidence, disease_label) == 1
        is_final_diagnosis_correct = get_disease_label_rank(final_diagnostic_confidence, disease_label) == 1

        for i in range(k):
            if disease_label in list(final_diagnostic_confidence.keys())[:i+1]:
                top_k_counts[i] += 1
        correct_wo_iq_count += 1 if is_initial_diagnosis_correct else 0
        correct_count += 1 if is_final_diagnosis_correct else 0
        turn_count += len(record["interactions"])
        consultation_round_count += count_consultation_rounds(record["interactions"])
    records_size = len(records)
    print(f"Acc_wo_iq: {round(correct_wo_iq_count / records_size, 3)}")
    print(f"Acc_iq: {round(correct_count / records_size, 3)}")
    print(f"Avg_n: {round(turn_count/ records_size, 1)}")
    print(f"Avg_consultation_rounds: {round(consultation_round_count / records_size, 1)}")
    print(f"Top-acc: {[round(top_k_counts[i] / records_size, 3) for i in range(k)]}")
    

def save_results(llm_name, dataset_name, exp_name, stage, timestep, results, settings={}):
    results["settings"] = settings
    results["retry"] = settings.get("retry", -1)
    results_dir = f"./outputs/policy/{dataset_name}/{exp_name}/"
    if stage == "dev":
        results_dir += f"checkpoints/{timestep}/"
    results_filepath = results_dir + f"results_{llm_name}_{timestep}_{settings.get('time', -1)}.json"

    with open(results_filepath, 'w', encoding="utf-8") as results_file:
        json.dump(results, results_file, ensure_ascii=False, indent=4)

def load_policy(dataset_name, exp_name, observation_space, action_space, lr_schedule, policy_kwargs):
    policy_path = f"./outputs/policy/{dataset_name}/{exp_name}/policy.pth"
    policy = SymptomInquiryActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lr_schedule,
        **policy_kwargs
    )
    policy.load_state_dict(th.load(policy_path), strict=False)
    return policy
        
        
def analysis_different_diseases(result_filepath):
    analysis_result_map = {}
    sub_records_map = {}
    with open(result_filepath, "r", encoding="utf-8") as result_file:
        records = json.load(result_file)["records"]
        
    min_n_map = {}
    max_candi_len = 0
    for record in records:
        disease_label = record["disease_label"]
        sub_records = sub_records_map.get(disease_label, [])
        sub_records.append(record) 
        sub_records_map[disease_label] = sub_records
        min_n = min(min_n_map.get(disease_label, 10), len(record["interactions"]))
        min_n_map[disease_label] = min_n
        for interaction in record["interactions"]:
            max_candi_len = max(max_candi_len, len(interaction["candidate_symptoms"]))
        
    for disease in sub_records_map.keys():
        analysis_result = calculate_metrics(sub_records_map[disease])
        analysis_result_map[disease] = {
            "Acc_wo_iq": analysis_result[0], 
            "Acc_iq": analysis_result[1], 
            "Avg_n": analysis_result[2], 
            "Avg_consultation_rounds": analysis_result[3],
            "min_n": min_n_map[disease_label],
            "max_candi_len": max_candi_len
        }
        
    print(analysis_result_map)
        
        
class MyEvalCallback(BaseCallback):
    def __init__(self, args, callback_interval, envs_dev = None, verbose: int = 1):
        super().__init__(verbose)
        self.args = args
        self.callback_interval = callback_interval
        self.eval_envs_dev = envs_dev
        self.best_metrics = {"Acc_wo_iq": 0, "Acc": 0, "Avg_n": 0}

    def _on_training_start(self) -> None:
        self.save_best_settings()
        
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        current_timestep = self.num_timesteps
        if current_timestep % self.callback_interval != 0 and current_timestep != 0:
            return True
        self.save_policy(f"./outputs/policy/{self.args.dataset_name}/{self.args.exp_name}/checkpoints/{current_timestep}")
        dev_metrics = performance_eval(self.args.llm_name, self.args.dataset_name, self.args.exp_name, "dev", current_timestep, self.eval_envs_dev, self.model.policy)
        
        if dev_metrics["Acc"] > self.best_metrics["Acc"]:
            self.best_timestep = current_timestep
            self.best_metrics["Acc"] = dev_metrics["Acc"]
            self.best_metrics["Avg_n"] = dev_metrics["Avg_n"]
            self.save_best_settings()
        return True
    
    def save_best_settings(self):
        exp_dir = f"./outputs/policy/{self.args.dataset_name}/{self.args.exp_name}"
        os.makedirs(exp_dir, exist_ok=True)
        best_settings = {"best_metrics": {"Acc": self.best_metrics["Acc"], "Avg_n": self.best_metrics["Avg_n"]}, "best_timestep": self.best_timestep, **vars(self.args)}
        self.save_policy(policy_model_path=exp_dir)
        with open(exp_dir + "/best_settings.json", "w", encoding="utf-8") as args_file:
            json.dump(best_settings, args_file, ensure_ascii=False, indent=4)
    
    def save_policy(self, policy_model_path):
        os.makedirs(policy_model_path, exist_ok=True)

        state_dict = self.model.policy.state_dict()

        for key in list(state_dict.keys()):
            for keyward in ['llm', 'tokenizer']:
                if keyward in key:
                    del state_dict[key]
                    break

        policy_path = policy_model_path + "/policy.pth"
        th.save(state_dict, policy_path)
    
def error_bars():
    import numpy as np
    from scipy import stats
    
    different_seed_results = {
        "DXY": [87.5, 84.6, 83.7, 82.7, 83.7],
        "dxy": [94.2, 91.3, 89.4, 93.3],
        "GMD": [79.5, 79.9, 79.5],
        "CMD": [63.6, 63.3, 62.4],
    }
        
    for dataset_name, data in different_seed_results.items():
        mean = np.mean(data)
        stderr = stats.sem(data)  # Standard error of the mean
        print(f"{dataset_name}: {mean:.1f} ± {stderr:.1f}")
    
    
    for dataset_name, data in different_seed_results.items():
        mean = np.mean(data)
        stderr = stats.sem(data) 
        ci95 = stderr * stats.t.ppf(0.975, len(data)-1)
        print(f"{dataset_name}: {mean:.2f} ± {ci95:.2f} (95% CI)")

def calc_f1(result_filepath):
    with open(result_filepath, "r", encoding="utf-8") as result_file:
        records = json.load(result_file)["records"]
    
    y_pred = []
    y_true = []
        
    if "DXY" in result_filepath:
        y_true.append("上呼吸道感染")
        y_pred.append("肺炎")   
    
    for record in records:
        y_true.append(record["disease_label"])
        y_pred.append(list(record["interactions"][-1]["diagnostic_confidence"].keys())[0])
        
    # 计算 Macro-F1 和 Micro-F1
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')

    print(f"{result_filepath}: Macro-F1: {macro_f1:.4f}, Micro-F1: {micro_f1:.4f}")


if __name__ == "__main__":
    result_filepath = ""
    calculate_metrics_offline(result_filepath)
    analysis_different_diseases(result_filepath)
    error_bars()
