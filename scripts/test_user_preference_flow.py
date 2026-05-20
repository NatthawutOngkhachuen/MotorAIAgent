from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.services.recommendation.langchain_slot_extractor import LangChainSlotExtractor
from app.services.recommendation.recommenders.user_based import UserBasedRecommender
from app.services.recommendation.slot_filling import SlotFillingService
from app.services.recommendation.vectorizer import FEATURE_NAMES, preference_to_vector, row_to_preference


class FakeLangChainExtractor:
    def __init__(self):
        self.calls = []

    def extract(
        self,
        *,
        user_message,
        chat_history,
        current_preferences,
        last_asked_slots,
        session_id,
    ):
        self.calls.append(
            {
                "user_message": user_message,
                "history_len": len(chat_history),
                "last_asked_slots": list(last_asked_slots),
                "session_id": session_id,
            }
        )
        text = user_message.lower()
        if "touring" in text:
            return {
                "budget_level": "high",
                "usage_fit": ["long_distance", "trip"],
                "style": ["premium", "modern"],
                "performance": "high",
                "comfort": "high",
                "safety_level": "high",
                "technology_level": "high",
                "easy_to_ride": False,
                "fuel_saving": False,
                "storage_need": True,
                "maintenance_easy": True,
            }
        if "delivery" in text:
            return {
                "budget_level": "medium",
                "usage_fit": ["delivery", "work", "city"],
                "style": ["compact"],
                "performance": "medium",
                "comfort": "medium",
                "safety_level": "medium",
                "technology_level": "medium",
                "easy_to_ride": True,
                "fuel_saving": True,
                "storage_need": True,
                "maintenance_easy": True,
            }
        if "long thai answer" in text:
            return {
                "usage_fit": ["work"],
                "style": ["beauty", "cute"],
                "fuel_saving": True,
                "maintenance_easy": True,
            }
        if "cheap" in text or "school" in text:
            return {
                "budget_level": "low",
                "usage_fit": ["daily", "city"],
                "style": ["sporty"],
            }
        if "comfortable" in text or "not too powerful" in text:
            return {"performance": "medium", "comfort": "high"}
        if "abs" in text or "smart key" in text:
            return {"safety_level": "high", "technology_level": "high"}
        if text.strip() in {"yes", "y"} and set(last_asked_slots) == {"easy_to_ride", "fuel_saving"}:
            return {"easy_to_ride": True, "fuel_saving": True}
        if "no storage" in text or "easy maintenance" in text:
            result = {"storage_need": False, "maintenance_easy": True}
            if "save fuel" in text:
                result["fuel_saving"] = True
            return result
        return {}


def test_vector_schema() -> None:
    assert len(FEATURE_NAMES) == 27

    pref = {
        "budget_level": "low",
        "usage_fit": ["city", "daily", "long_distance"],
        "style": ["sporty", "premium"],
        "performance": "high",
        "comfort": "unknown",
        "safety_level": "medium",
        "technology_level": "high",
        "easy_to_ride": True,
        "fuel_saving": True,
        "storage_need": False,
        "maintenance_easy": "unknown",
    }
    vector = preference_to_vector(pref)
    assert len(vector) == 27
    assert vector[FEATURE_NAMES.index("usage_fit_city")] == 1.0
    assert vector[FEATURE_NAMES.index("usage_fit_daily")] == 1.0
    assert vector[FEATURE_NAMES.index("style_sporty")] == 1.0
    assert vector[FEATURE_NAMES.index("style_premium")] == 1.0
    assert vector[FEATURE_NAMES.index("budget_level_score")] == 0.33
    assert vector[FEATURE_NAMES.index("comfort_score")] == 0.0
    assert vector[FEATURE_NAMES.index("storage_need_flag")] == 0.0


def test_row_unknowns_do_not_become_features() -> None:
    pref = row_to_preference(
        {
            "budget_level": "Unknown",
            "usage_fit": "Unknown",
            "style": "sporty,beauty",
            "performance": "Unknown",
            "comfort": "medium",
            "safety_level": "Unknown",
            "technology_level": "high",
            "easy_to_ride": "TRUE",
            "fuel_saving": "Unknown",
            "storage_need": "Unknown",
            "maintenance_easy": "TRUE",
        }
    )
    vector = preference_to_vector(pref)
    assert vector[FEATURE_NAMES.index("usage_fit_city")] == 0.0
    assert vector[FEATURE_NAMES.index("style_sporty")] == 1.0
    assert vector[FEATURE_NAMES.index("style_beauty")] == 1.0
    assert vector[FEATURE_NAMES.index("fuel_saving_flag")] == 0.0


def test_slot_filling_many_answer_styles() -> None:
    service = SlotFillingService()
    first_question, state = service.start()
    assert "ใช้รถ" in first_question

    state, question = service.handle_message("ไม่แพงมาก ใช้ไปเรียนทุกวันในเมือง ชอบทรงสปอร์ตเท่ๆ", state)
    assert state.preferences["budget_level"] == "unknown"
    assert "daily" in state.preferences["usage_fit"]
    assert "city" in state.preferences["usage_fit"]
    assert "sporty" in state.preferences["style"]
    assert "เน้นความเร็ว" in question

    state, question = service.handle_message("ขี่ทั่วไป", state)
    assert state.preferences["performance"] == "medium"
    assert "สบาย" in question

    state, question = service.handle_message("มาก", state)
    assert state.preferences["comfort"] == "high"
    assert "ปลอดภัย" in question

    state, question = service.handle_message("ปลอดภัยสูง ฟีเจอร์เยอะ", state)
    assert state.preferences["safety_level"] == "high"
    assert state.preferences["technology_level"] == "unknown"
    assert "ฟังก์ชัน" in question

    state, question = service.handle_message("1 2", state)
    assert state.preferences["easy_to_ride"] is True
    assert state.preferences["fuel_saving"] is True
    assert question is None
    assert state.is_complete is True
    assert state.preferences["storage_need"] == "unknown"
    assert state.preferences["maintenance_easy"] == "unknown"
    vector = service.build_vector(state.preferences)
    assert vector[FEATURE_NAMES.index("budget_level_score")] == 0.0
    assert vector[FEATURE_NAMES.index("technology_level_score")] == 0.0
    assert vector[FEATURE_NAMES.index("storage_need_flag")] == 0.0
    assert vector[FEATURE_NAMES.index("maintenance_easy_flag")] == 0.0


def test_slot_filling_follow_up_restart_question() -> None:
    service = SlotFillingService()
    question, state = service.start_follow_up()

    assert question.startswith("สนใจดูรถแนวไหนเพิ่มเติมอีกมั้ยครับ")
    assert "ปกติใช้รถทำอะไรเป็นหลัก" in question
    assert state.preferences["usage_fit"] == []
    assert state.asked_slots == ["usage_fit"]
    assert state.last_asked_slots == ["usage_fit"]
    assert state.is_complete is False


def test_long_answer_skips_slots_already_extracted() -> None:
    extractor = FakeLangChainExtractor()
    service = SlotFillingService(extractor=extractor)
    _, state = service.start()

    state, question = service.handle_message(
        "long thai answer อยากได้รถที่ขี่สบาย ทรงสวยน่ารัก เอาไว้ขี่ไปทำงาน ประหยัดน้ำมัน ดูแลง่าย",
        state,
        chat_history=[{"role": "assistant", "content": "usage question"}],
        session_id="long-answer",
    )

    assert "work" in state.preferences["usage_fit"]
    assert "beauty" in state.preferences["style"]
    assert state.preferences["fuel_saving"] is True
    assert state.preferences["maintenance_easy"] is True
    assert state.last_asked_slots == ["performance"]
    assert "ฟีลการขี่" in question


def test_langchain_memory_chat_flow_three_rounds() -> None:
    # Round 1: multi-turn normal flow with short yes/no answer interpreted by memory.
    extractor = FakeLangChainExtractor()
    service = SlotFillingService(extractor=extractor)
    _, state = service.start()
    history = [{"role": "assistant", "content": "usage question"}]

    state, question = service.handle_message(
        "cheap bike for school in the city, sporty look",
        state,
        chat_history=history,
        session_id="round-1",
    )
    history.extend(
        [
            {"role": "user", "content": "cheap bike for school in the city, sporty look"},
            {"role": "assistant", "content": question or ""},
        ]
    )
    state, question = service.handle_message("comfortable, not too powerful", state, history, "round-1")
    history.extend(
        [
            {"role": "user", "content": "comfortable, not too powerful"},
            {"role": "assistant", "content": question or ""},
        ]
    )
    state, question = service.handle_message("ขี่ทั่วไป", state, history, "round-1")
    history.extend(
        [
            {"role": "user", "content": "ขี่ทั่วไป"},
            {"role": "assistant", "content": question or ""},
        ]
    )
    state, question = service.handle_message("มาก", state, history, "round-1")
    history.extend(
        [
            {"role": "user", "content": "มาก"},
            {"role": "assistant", "content": question or ""},
        ]
    )
    state, question = service.handle_message("สูง", state, history, "round-1")
    history.extend(
        [
            {"role": "user", "content": "สูง"},
            {"role": "assistant", "content": question or ""},
        ]
    )
    state, question = service.handle_message("1 2 4", state, history, "round-1")
    assert state.is_complete is True
    assert question is None
    assert state.preferences["easy_to_ride"] is True
    assert state.preferences["fuel_saving"] is True
    assert state.preferences["storage_need"] == "unknown"
    assert extractor.calls[-1]["history_len"] > 0

    # Round 2: user answers many slots at once. The service should not ask repeats.
    extractor2 = FakeLangChainExtractor()
    service2 = SlotFillingService(extractor=extractor2)
    _, state2 = service2.start()
    state2, question2 = service2.handle_message(
        "touring, premium modern, high budget, high safety, smart key, storage, easy maintenance",
        state2,
        chat_history=[{"role": "assistant", "content": "usage question"}],
        session_id="round-2",
    )
    assert state2.is_complete is True
    assert question2 is None
    assert state2.preferences["budget_level"] == "unknown"
    assert state2.preferences["technology_level"] == "unknown"
    assert state2.preferences["maintenance_easy"] is True
    assert "long_distance" in state2.preferences["usage_fit"]

    # Round 3: work/delivery user with compact/practical needs completes in one response.
    extractor3 = FakeLangChainExtractor()
    service3 = SlotFillingService(extractor=extractor3)
    _, state3 = service3.start()
    state3, question3 = service3.handle_message(
        "delivery work in city, medium budget, compact, save fuel, easy maintenance, storage",
        state3,
        chat_history=[{"role": "assistant", "content": "usage question"}],
        session_id="round-3",
    )
    assert state3.is_complete is True
    assert question3 is None
    assert "delivery" in state3.preferences["usage_fit"]
    assert state3.preferences["fuel_saving"] is True


def test_langchain_runnable_with_message_history_extractor() -> None:
    extractor = LangChainSlotExtractor()
    extractor.llm = FakeListChatModel(
        responses=[
            '{"easy_to_ride": true, "fuel_saving": true}',
        ]
    )
    result = extractor.extract(
        user_message="yes",
        chat_history=[
            {
                "role": "assistant",
                "content": "อยากได้รถที่ขี่ง่ายและประหยัดน้ำมันเป็นพิเศษไหมครับ?",
            }
        ],
        current_preferences={"usage_fit": ["city"], "style": ["sporty"]},
        last_asked_slots=["easy_to_ride", "fuel_saving"],
        session_id="fake-history-test",
    )
    assert result["easy_to_ride"] is True
    assert result["fuel_saving"] is True


def test_recommender_outputs() -> None:
    recommender = UserBasedRecommender()
    pref = {
        "budget_level": "low",
        "usage_fit": ["city", "daily"],
        "style": ["sporty"],
        "performance": "medium",
        "comfort": "medium",
        "safety_level": "medium",
        "technology_level": "medium",
        "easy_to_ride": True,
        "fuel_saving": True,
        "storage_need": False,
        "maintenance_easy": True,
    }
    nearest = recommender.recommend_nearest_user(pref, top_k=3)
    cluster = recommender.recommend_cluster(pref, top_k=5)
    assert nearest["matched_user_id"]
    assert len(nearest["candidates"]) >= 1
    assert cluster["cluster"] in {0, 1, 2}
    assert len(cluster["candidates"]) >= 1


def test_generated_assets_shape() -> None:
    vectors = pd.read_csv(PROJECT_ROOT / "data/recommendation_output/final/user_preference_vectors_expanded.csv")
    feature_map = pd.read_csv(PROJECT_ROOT / "data/recommendation_output/final/user_preference_feature_map.csv")
    centroids = pd.read_csv(PROJECT_ROOT / "data/recommendation_output/final/user_preference_cluster_centroids.csv")
    assert len(vectors) == 91
    assert len(feature_map) == 27
    assert set(FEATURE_NAMES).issubset(vectors.columns)
    assert set(FEATURE_NAMES).issubset(centroids.columns)


def main() -> None:
    tests = [
        test_vector_schema,
        test_row_unknowns_do_not_become_features,
        test_slot_filling_many_answer_styles,
        test_slot_filling_follow_up_restart_question,
        test_long_answer_skips_slots_already_extracted,
        test_langchain_memory_chat_flow_three_rounds,
        test_langchain_runnable_with_message_history_extractor,
        test_recommender_outputs,
        test_generated_assets_shape,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
