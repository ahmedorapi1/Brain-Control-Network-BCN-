import numpy as np
import onnxruntime as ort

model_path = r"D:\GP_data\EEG\digit_model\over\digits-model.onnx"

session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

data = np.load(r"D:\GP_data\EEG\digit_model\over\4.npy")
X = data[0]

x_input = np.expand_dims(X, axis=0).astype(np.float32)

outputs = session.run(
    [output_name],
    {input_name: x_input}
)

pred = outputs[0]

pred_class = np.argmax(pred, axis=1)[0]

print("Pred Label :", pred_class)
