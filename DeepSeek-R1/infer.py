import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import unicodedata
from typing import List

@torch.inference_mode()
def generate(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0
) -> List[int]:
    """
    Generate response from the model with attention_mask provided.
    """
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,  # 提供显式 attention mask
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    return outputs[0].tolist()

def clean_input(user_input):
    """
    清理用户输入，去除不可见字符和多余的空格。
    """
    user_input = "".join(c for c in user_input if not unicodedata.category(c).startswith("C"))  # 移除控制字符
    return user_input.strip()  # 去除首尾空格

def clean_message_content(content):
    """
    清理消息内容，去除首尾空格并过滤非法输入
    """
    if not content or not isinstance(content, str):
        return ""
    return content.strip()  # 去除首尾空格

def build_prompt(messages, max_history=3):
    """
    Build prompt for the model, limiting the history to the most recent messages.
    """
    template = "The following is a conversation with an AI assistant. The assistant is helpful, knowledgeable, and polite:\n"
    for msg in messages[-max_history:]:
        content = clean_message_content(msg["content"])
        if not content:  # 跳过空内容
            continue
        template += f"{msg['role'].capitalize()}: {content}\n"
    template += "Assistant: "
    return template.strip()  # 确保返回值是字符串

if __name__ == "__main__":
    print("Initializing DeepSeek-R1 Service...")

    # Configuration
    ckpt_path = "./"  # 模型所在的目录
    config_path = "./config.json"  # 配置文件路径

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
    ).cuda()

    # Interactive session
    messages = []  # To maintain context
    while True:
        user_input = input("You: ").strip()  # 去除首尾空格
        user_input = clean_input(user_input)  # 清理不可见字符
        if not user_input or len(user_input.strip()) == 0:  # 检查无效输入
            print("Invalid input. Please type something meaningful!")
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting conversation. Goodbye!")
            break

        # Append user input to context
        messages.append({"role": "user", "content": user_input})

        # Limit conversation history
        messages = messages[-10:]  # 只保留最近 10 条对话

        # Build prompt and tokenize
        prompt = build_prompt(messages)
        if not isinstance(prompt, str) or len(prompt.strip()) == 0:  # 确保 prompt 非空
            print("Error: Prompt is empty or invalid. Skipping this turn.")
            continue

        tokenized_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        input_ids = tokenized_prompt["input_ids"].to("cuda")
        attention_mask = tokenized_prompt["attention_mask"].to("cuda")

        # Generate response
        max_new_tokens = 150
        temperature = 0.7

        completion_tokens = generate(model, input_ids, attention_mask, max_new_tokens, temperature)
        completion = tokenizer.decode(
            completion_tokens[len(input_ids[0]):],  # 从输入长度截取生成部分
            skip_special_tokens=True
        ).split("User:")[0].strip()

        print(f"Assistant: {completion}")

        # Append assistant response to context
        messages.append({"role": "assistant", "content": completion})
