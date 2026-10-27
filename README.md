# GPT-2 Custom Fine-Tuner

A lightweight Streamlit web application that allows users to fine-tune the Hugging Face GPT-2 language model on custom text datasets. 

## Features
* **Frontend UI:** Built with Streamlit for easy interaction.
* **Custom Dataset Upload:** Upload any `.txt` file to train the model on specific vocabulary, tone, or context.
* **Hyperparameter Controls:** Adjustable sliders for training Epochs and Batch Size.
* **Local Training:** Utilizes PyTorch and Hugging Face `transformers` to execute the training loop locally.
* **Side-by-Side Comparison:** Test prompts against both the Base GPT-2 model and the newly Fine-Tuned model to observe the diagnostic differences.

## Installation

1. Clone this repository:
   ```bash
   git clone [https://github.com/AryanSovit292/gpt2-finetuner.git](https://github.com/AryanSovit292/gpt2-finetuner.git)
   cd gpt2-finetuner