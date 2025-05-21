from transformers import AutoModelForCausalLM
from datasets import load_dataset
from episode_generator.sports_episode_generator import SportsEpisodeGenerator
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch
import matplotlib.pyplot as plt

def ppo_clipped_loss(log_probs_old, log_probs_new, reward, eps=0.2):
    ratio = torch.exp(log_probs_new - log_probs_old)
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
    return -torch.mean(torch.min(ratio * reward, clipped * reward))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    generator = SportsEpisodeGenerator(model)
    optimizer = Adam(model.parameters(), lr=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

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

        # Safe normalization or clipping
        if reward_tensor.numel() > 1:
            reward_tensor = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-8)
        else:
            reward_tensor = torch.clamp(reward_tensor, 0.0, 1.0)

        # PPO clipped loss
        loss = ppo_clipped_loss(old_lp_sum.detach(), new_lp_sum, reward_tensor)

        if torch.isnan(loss):
            print(f"Skipping step {step} due to NaN loss")
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        print(f"Step {step} | Reward: {episode.reward:.2f} | Loss: {loss.item():.4f}")

        rewards.append(episode.reward)
        losses.append(loss.item())

        if step > 200:  # Extended training steps
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
