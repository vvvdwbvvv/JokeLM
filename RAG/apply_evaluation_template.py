import random
import os

TEMPLATE = "幫我寫一個和{topic}有關的笑話"

def load_topics(filename):
    """Loads topics from a file, filtering out empty lines and markers."""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('---')]

def generate_prompts(template, topics):
    """Generates a list of prompts from a template and a list of topics."""
    return [template.format(topic=topic) for topic in topics]

def save_topics(filename, topics):
    """Saves a list of topics to a file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for topic in topics:
            f.write(f"{topic}\n")

def main():
    # Define file paths relative to the script location for better portability
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data', 'eval')
    
    input_file = os.path.join(data_dir, 'topics.txt')
    eval_file = os.path.join(data_dir, 'eval_topics.txt')
    gen_file = os.path.join(data_dir, 'gen_topics.txt')

    # 1. Load topics ONCE
    all_topics = load_topics(input_file)
    
    # 2. Shuffle and split the list of topics
    random.shuffle(all_topics)
    split_ratio = 0.8
    split_index = int(len(all_topics) * split_ratio)
    
    eval_topics = all_topics[:split_index]
    gen_topics = all_topics[split_index:]
    
    # 3. Save the split topic lists to their respective files
    save_topics(eval_file, eval_topics)
    save_topics(gen_file, gen_topics)
    
    print(f"Successfully split {len(all_topics)} topics.")
    print(f"  - {len(eval_topics)} topics saved to {eval_file}")
    print(f"  - {len(gen_topics)} topics saved to {gen_file}")
    print("-" * 20)

    # 4. Generate prompts from the SPLIT lists
    eval_prompts = generate_prompts(TEMPLATE, eval_topics)
    gen_prompts = generate_prompts(TEMPLATE, gen_topics)

    # 5. Print a few examples from each set
    print("--- Example Evaluation Prompts ---")
    for prompt in eval_prompts[:3]:
        print(prompt)
        
    print("\n--- Example Generation Prompts ---")
    for prompt in gen_prompts[:3]:
        print(prompt)

if __name__ == "__main__":
    main()