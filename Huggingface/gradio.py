import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoImageProcessor, ResNetForImageClassification
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from PIL import Image, ImageDraw,ImageFont

token = "####" # group token

def ObjectDetection(image):
    image = Image.fromarray(image)
    processor = DetrImageProcessor.from_pretrained("ailab/detr-resnet-50", token="####",revision="main")
    model = DetrForObjectDetection.from_pretrained("ailab/detr-resnet-50", token="####",revision="main")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    res = []
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 12)  # 替换为适合的字体和字号
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        res.append(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
        x1, y1, x2, y2 = box
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
        draw.text((x1, y1 - 15), model.config.id2label[label.item()], fill="black", font=font)  # 调整文字位置和颜色
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
    return  image


def classifyImage(image):
    # processor = AutoImageProcessor.from_pretrained("wyq/resnet-50",token="####")
    # model = ResNetForImageClassification.from_pretrained("wyq/resnet-50",token="####")
    processor = AutoImageProcessor.from_pretrained("ailab/resnet-50",token="####",revision="master")
    model = ResNetForImageClassification.from_pretrained("ailab/resnet-50",token="####",revision="master")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    print(model.config.id2label[predicted_label])
    return  model.config.id2label[predicted_label]


def  TextSentimentAnalysis(text):
    tokenizer = AutoTokenizer.from_pretrained("wyq/twitter-roberta-base-sentiment-latest",token="####",revision="main")
    config = AutoConfig.from_pretrained("wyq/twitter-roberta-base-sentiment-latest",token="####",revision="main")
    model = AutoModelForSequenceClassification.from_pretrained("wyq/twitter-roberta-base-sentiment-latest",token="####",revision="main")
    # text = "Covid cases are increasing fast!"
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    res = {}
    for i in range(scores.shape[0]):
        l = config.id2label[ranking[i]]
        s = scores[ranking[i]]
        res[l] = np.round(float(s), 4)
        print(f"{i+1}) {l} {np.round(float(s), 4)}")
    return res


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def ImageToText(image):
    model = VisionEncoderDecoderModel.from_pretrained("ailab/vit-gpt2-image-captioning",token="####",revision="main")
    feature_extractor = ViTImageProcessor.from_pretrained("ailab/vit-gpt2-image-captioning",token="####",revision="main")
    tokenizer = AutoTokenizer.from_pretrained("ailab/vit-gpt2-image-captioning",token="####",revision="main")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    images = []
    images.append(image)
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    print(preds)
    return preds



def GenerateText(prompt,history): 
    model = AutoModelForCausalLM.from_pretrained(
        "ailab/Qwen1.5-1.8B-Chat",
        torch_dtype="auto",
        device_map="auto",
        token="####"
    )
    tokenizer = AutoTokenizer.from_pretrained("ailab/Qwen1.5-1.8B-Chat", token="####")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt")

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    history.append((prompt,response))
    return "", history


with gr.Blocks() as demo:
    gr.Markdown("DEMO.")
    with gr.Tab("ImageToText"):
        with gr.Column():
            text1_input = gr.Image()
            text1_output = gr.Textbox()
            text1_button = gr.Button("Submmit")
    with gr.Tab("generateText"):
        with gr.Column():
            text_input = gr.Textbox()
            text_output = gr.Chatbot()
            text2_button = gr.Button("Submmit")
            clear = gr.ClearButton([text_input, text_output]) 
    with gr.Tab("objectDetection"):
        with gr.Row():  
            image_input = gr.Image()
            image_output = gr.Image()
        image_button = gr.Button("Submmit")
    with gr.Tab("classifyImage"):
        with gr.Row():
            image2_input = gr.Image()
            text2_output = gr.Textbox()
        image2_button = gr.Button("Submmit")
    # with gr.Accordion("Open for More!"):
    #     gr.Markdown("Look at me...")
    text1_button.click(ImageToText, inputs=text1_input, outputs=text1_output)
    text2_button.click(GenerateText, inputs=[text_input,text_output], outputs=[text_input,text_output])
    image_button.click(ObjectDetection, inputs=image_input, outputs=image_output)
    image2_button.click(classifyImage, inputs=image2_input, outputs=text2_output)
demo.launch()


