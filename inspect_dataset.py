import random

from datasets import load_dataset


dataset = load_dataset("Parveshiiii/AI-vs-Real", verification_mode="no_checks")
label_map = {0: "AI-generated", 1: "Real"}
indices = random.sample(range(len(dataset["train"])), 100)

for i in indices:
    example = dataset["train"][i]
    if example["binary_label"] == 0:
        print("AI-generated")
        example["image"].show()
