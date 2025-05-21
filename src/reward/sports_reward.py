from bert_score import BERTScorer

class SportsRankedReward:
    def __init__(self, lang="en"):
        model_name = "roberta-large"
        self.scorer = BERTScorer(model_type=model_name, lang=lang, rescale_with_baseline=True)

    def __call__(self, prompt: str, response: str, instance: dict) -> float:
        candidates = instance["candidates"]
        ranking = instance["ranking"]
        reference = candidates[ranking[0]].strip()

        P, R, F1 = self.scorer.score([response], [reference])
        return F1[0].item()
