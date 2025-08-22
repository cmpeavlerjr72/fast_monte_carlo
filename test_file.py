import pandas as pd
from pass_outcome_infer import PassOutcomeTwoStage

model = PassOutcomeTwoStage()

row = {
    "down": 3, "distance": 7, "yardsToGoal": 35, "is_red_zone": 0,
    "score_diff": -3, "seconds_remaining": 742,
    "offenseTimeouts": 2, "defenseTimeouts": 2,
    "sp_rating_off": 12.0, "sp_offense_rating_off": 18.0,
    "sp_defense_rating_def": 10.0, "sp_rating_def": 7.0,
    "goal_to_go": 0, "fourth_and_short": 0, "fg_range": 0, "half": 2, "two_minute": 0,
    "passer_name": "Caleb Williams",
    "target_name": "Unknown"  # ok if you don't know pre-snap
}
x = pd.DataFrame([row])
probs = model.predict_proba(x)[0]  # [p_complete, p_incomplete, p_intercepted, p_sack]
print(dict(zip(["complete","incomplete","intercepted","sack"], probs)))
