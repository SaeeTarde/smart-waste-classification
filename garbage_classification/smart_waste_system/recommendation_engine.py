#recommendation_engine.py
# This function bridges AI output (waste class)
# with sustainability intelligence (human-defined rules)

from .knowledge_base import WASTE_RULES

def recommend(waste_type):
    if waste_type not in WASTE_RULES:
        return {
            "disposal_method":"Unknown",
            "eco_score":0,
            "tip":"No recommendation available"
        }
    return WASTE_RULES[waste_type]