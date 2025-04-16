from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import textwrap
from multiprocessing import Process, Queue, Event
from multiprocess_inference import llm_process,  vision_encoder_process

# Initialize Model processes
load_ready_queue = Queue()
embedding_queue = Queue()
img_path_queue = Queue()
prompt_queue = Queue()
inference_done_queue = Queue()
start_event = Event()

vision_process = Process(target=vision_encoder_process,
                        args=(load_ready_queue, embedding_queue, img_path_queue, start_event))
lm_process = Process(target=llm_process,
                    args=(load_ready_queue, embedding_queue, prompt_queue, inference_done_queue, start_event))

vision_process.start()
lm_process.start()

ready_count = 0
while ready_count < 2:
    status = load_ready_queue.get()
    print(f"Received ready signal: {status}")
    ready_count += 1

print("All models loaded, starting interactive mode...")
start_event.set()

# Flask app
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

image_placeholder = '<image_id>0</image_id><image>\n' 
def chunk_text(text):
    """Chunk text into smaller pieces."""
    chunks = textwrap.wrap(text, width=21)

    merged_text = "\n".join(chunk + image_placeholder for chunk in chunks)

    prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{merged_text}<|im_end|>
<|im_start|>assistant
"""
    return prompt


@app.route('/upload', methods=['POST'])
def upload_file():
    text = request.form.get("text")  
    image = request.files.get("image")  

    if not text or not image:
        return jsonify({"error": "Text and image are required"}), 400

    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    prompt = chunk_text(text)

    img_path_queue.put(image_path)
    prompt_queue.put(prompt)

    result = inference_done_queue.get()

    if result == "ERROR":
        return jsonify({"error": "Inference failed"}), 500
    
    if isinstance(result, list):
        result = "".join(result)

    return jsonify({
        "text_received": text,
        "image_path": image_path,
        "result": result 
    })

if __name__ == '__main__':
    app.run(debug=False, port=8080, use_reloader=False)
