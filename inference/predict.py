from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType
import matplotlib.pyplot as plt
import cv2

model = YOLO(r"D:\GP_data\vision_data\best.pt")

results = model.predict(
    source="test2.jpg",
    conf=0.4,
    show=False
)
img = results[0].plot()

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

quantize_dynamic(                      #int8
    "best.onnx",
    "best_int8.onnx",
    weight_type=QuantType.QInt8
)

model.export(                      #fp32
    format="onnx",
    opset=12,
    simplify=True,
    dynamic=True
)