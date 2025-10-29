
# Building a Small Language Model (SLM) (for dummies)

This repository contains the complete code for building, training, and testing a small GPT-style language model from the ground up using PyTorch. The goal of this project was to demystify the "black box" of LLMs by writing every component manually, from the data loading to the training loop.

The model is trained on the **TinyStories dataset** to generate simple, creative stories.

## Project Goal

The primary goal was to build a small-scale (10-60M parameters) transformer model to understand every component involved in its creation. We successfully built a **\~10.7M parameter model** that learns the fundamentals of the English language.

## Architecture

![arch](https://file.notion.so/f/f/cc2e367d-8a55-498e-a069-135390335a78/136da529-4cdf-4045-a457-4e9a35e887dc/SLM_Architecture.webp?table=block&id=29bd5368-c6db-8014-9140-f79b1686e939&spaceId=cc2e367d-8a55-498e-a069-135390335a78&expirationTimestamp=1761748626445&signature=ZhDJny5MpXPBCJgfcJc0D3w5L5uMZNWaN5GffRZwVjE&downloadName=SLM_Architecture.webp)

The model is a custom GPT-style transformer, built using pure PyTorch. The key components, defined in the script, are:

* **`GPT`**: The main model class that assembles the components.

* **`Block`**: A single Transformer block, containing:

  * **`CausalSelfAttention`**: A multi-head causal self-attention layer (with a manual mask fallback if Flash Attention isn't available).

  * **`MLP`**: A simple two-layer feed-forward network.

* **Embeddings**: Separate token (`wte`) and positional (`wpe`) embeddings.

* **Weight Tying**: The token embedding weights are tied to the final `lm_head` for efficiency.

You can also see a complete technical diagram in Mermaid syntax in `SLM_Architecture_Diagram.md`.

## Key Features

* **Data Processing**: Uses `tiktoken` for GPT-2 encoding and `np.memmap` to stream the massive dataset from disk, avoiding high RAM usage (inspired by nanoGPT).

* **Custom Training Loop**: No `transformers.Trainer`! The training loop is written from scratch, manually handling:

  * Batch fetching.

  * Mixed-precision training (`torch.amp.autocast`).

  * Gradient scaling (`GradScaler`).

  * Gradient accumulation (to simulate a large batch size).

  * Gradient clipping.

* **Optimization**: Uses the **AdamW** optimizer with `weight_decay`.

* **Learning Rate Schedule**: Implements a `LinearLR` warmup followed by a `CosineAnnealingLR` decay for stable training.

## How to Run

1. **Install Dependencies:**

```

pip install transformers datasets accelerate tiktoken torch

```

2. **Run the Training Script:**
Execute the main Python script (e.g., in a Colab notebook or as a `.py` file).

```

python Finetuning\_SLM\_for\_Text\_Transformation.py

```

The script will automatically:

1. Download and process the `roneneldan/TinyStories` dataset.

2. Create `train.bin` and `validation.bin` files.

3. Start the training loop for 20,000 iterations.

4. Evaluate the model every 500 steps and save the best-performing weights to `best_model_params.pt`.

5. Plot the training and validation loss curves.

6. Run inference on two example prompts using the best saved model.

## Results & Analysis

The model successfully learns the foundations of language! It understands grammar, punctuation, and subject-verb agreement.

However, as a small model with a limited 128-token context window, its "logic" is that of an **"Alien Toddler."**

* **Locally Coherent:** Sentences make grammatical sense.

* **Globally Drifting:** It cannot hold a coherent plot. The story "drifts" from topic to topic (e.g., from a pumpkin to an ant to a worm).

* **Creative Hallucinations:** It invents words ("hisding," "igibrarian") and makes charmingly strange associations ("...a big, hairy lime - it was painful, like a carrot!").
