import numpy as np
import onnxruntime as ort

model_path = r"D:\GP_data\EEG\digit_model\over\digits-model.onnx"

session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

data = np.load(r"D:\GP_data\EEG\digit_model\over\sampled_correct_predictions.npz")
X = data["X"]
y = data["y"]

idx = 22

x_sample = X[idx]
y_true = y[idx]

x_input = np.expand_dims(x_sample, axis=0).astype(np.float32)

outputs = session.run(
    [output_name],
    {input_name: x_input}
)

pred = outputs[0]

pred_class = np.argmax(pred, axis=1)[0]

print("True Label :", y_true)
print("Pred Label :", pred_class)

if pred_class == y_true:
    print("Correct Prediction")
else:
    print("Wrong Prediction")