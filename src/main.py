from transformers import AutoModelForCausalLM
from datasets import load_dataset
from episode_generator.sports_episode_generator import SportsEpisodeGenerator
from torch.optim import Adam
import torch
import matplotlib.pyplot as plt

def policy_loss(log_probs_old, log_probs_new, reward):
    ratio = torch.exp(log_probs_new - log_probs_old)
    return -torch.mean(ratio * reward)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    generator = SportsEpisodeGenerator(model)
    optimizer = Adam(model.parameters(), lr=1e-5)

    dataset = load_dataset("json", data_files="data/combined_dpo_samples_sorted.jsonl")['train']

    rewards = []
    losses = []

    for step, instance in enumerate(dataset):
        episode = generator.generate_episode(instance)

        # Combine prompt and response tokens for full input
        inputs = torch.cat([episode.query_tensor, episode.response_tensor]).unsqueeze(0).to(device)

        # Old log probs (no grad)
        with torch.no_grad():
            old_outputs = model(inputs)
        old_log_probs = torch.log_softmax(old_outputs.logits[0, :-1], dim=-1)
        old_lp = old_log_probs[torch.arange(old_log_probs.size(0)), inputs[0, 1:]]
        old_lp_sum = old_lp.sum()

        # New log probs (with grad)
        outputs = model(inputs)
        log_probs = torch.log_softmax(outputs.logits[0, :-1], dim=-1)
        new_lp = log_probs[torch.arange(log_probs.size(0)), inputs[0, 1:]]
        new_lp_sum = new_lp.sum()

        reward_tensor = torch.tensor(episode.reward).to(device)
        loss = policy_loss(old_lp_sum.detach(), new_lp_sum, reward_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step} | Reward: {episode.reward:.2f} | Loss: {loss.item():.4f}")

        rewards.append(episode.reward)
        losses.append(loss.item())

        if step > 20:  # change to higher number for full training
            break

    # Plot reward curve
    plt.plot(rewards)
    plt.title("Reward Over Steps")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.savefig("reward_curve.png")
    plt.clf()

    # Plot loss curve
    plt.plot(losses)
    plt.title("Loss Over Steps")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig("loss_curve.png")
    plt.clf()

if __name__ == "__main__":
    main()
