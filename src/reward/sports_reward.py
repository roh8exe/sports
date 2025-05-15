from fuzzywuzzy import fuzz

class SportsRankedReward:
    def __call__(self, prompt: str, response: str, instance: dict) -> float:
        candidates = instance["candidates"]
        ranking = instance["ranking"]
        
        top = candidates[ranking[0]].strip().lower()
        res = response.strip().lower()

        # Fuzzy partial match: returns score between 0 and 100
        score = fuzz.partial_ratio(res, top)
        return score / 100.0  # Normalize to [0, 1]
