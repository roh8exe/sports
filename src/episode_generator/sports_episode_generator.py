from transformers import AutoTokenizer
from episode_generator.episode import Episode
from reward.sports_reward import SportsRankedReward
import torch

class SportsEpisodeGenerator:
    def __init__(self, model, tokenizer_path="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = model
        self.reward_fn = SportsRankedReward()

    def build_prompt(self, instance):
        return "Summarize the match in one sentence."

    def generate_episode(self, instance):
        prompt = self.build_prompt(instance)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        reward = self.reward_fn(prompt, response, instance)

        return Episode(inputs["input_ids"].squeeze(0), response_ids, reward, response)
